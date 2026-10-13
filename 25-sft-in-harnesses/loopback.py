# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Watch which network connections a harness opens, with ``lsof``, while it runs.

Chapter 19's sandbox already refuses every connection that does not go to
this machine. This is the other half of the proof: look at what the harness
actually did. Every tenth of a second, while the run lasts, ``lsof`` lists the
internet sockets of the harness and of every process it started: those
below it by parent id (a shell the model drives is a child of a child), and
those in its process group, which a background command keeps even when its
parent exits. Each socket is either connected (``local->remote``) or open
with no peer (listening, or a UDP socket nothing was sent on). A
connection counts as loopback when its remote end is a loopback address
(``127.0.0.1``, ``::1``, ``localhost``).

A connection that opens and closes between two samples is missed, and so is
a process that leaves both the tree and the group, so the watch can say
"nothing else was seen", never "nothing else happened"; the sandbox's rule
is what makes the second true. A sample that could not be taken is counted,
and a watch that lost any sample does not count as loopback only. Only
Python's standard library and macOS's own ``lsof`` and ``ps`` are needed.
"""

from __future__ import annotations

import ipaddress
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

LSOF = "/usr/sbin/lsof"
PS = "/bin/ps"
INTERVAL_S = 0.1
COMMAND_TIMEOUT_S = 10
ARROW = "->"
LOCALHOST = "localhost"
PS_FIELDS = 3  # pid, parent id, process group id


@dataclass
class NetworkReport:
    """What the samples saw.

    Attributes:
        samples: How many times ``lsof`` looked.
        failed: Samples that could not be taken (``ps`` or ``lsof`` failed).
        seconds: From the first sample to the last.
        connections: Each remote end seen, with the programs that held it.
        unconnected: Sockets with no remote end (listening or unused), by
            local address.
    """

    samples: int = 0
    failed: int = 0
    seconds: float = 0.0
    connections: dict[str, set[str]] = field(default_factory=dict)
    unconnected: dict[str, set[str]] = field(default_factory=dict)

    @property
    def outside(self) -> list[str]:
        """Remote ends that are not this machine."""
        return sorted(r for r in self.connections if not is_loopback(r))

    @property
    def loopback_only(self) -> bool:
        """True when the watch looked every time and saw only this machine."""
        return self.samples > 0 and self.failed == 0 and not self.outside

    def as_dict(self) -> dict[str, object]:
        """Write the report as JSON-ready data.

        Returns:
            The report.
        """
        return {
            "samples": self.samples,
            "failed_samples": self.failed,
            "seconds": round(self.seconds, 1),
            "connections": {k: sorted(v) for k, v in sorted(self.connections.items())},
            "unconnected": {k: sorted(v) for k, v in sorted(self.unconnected.items())},
            "loopback_only": self.loopback_only,
        }


def host_of(address: str) -> str:
    """Take the host out of ``host:port`` (IPv6 hosts come in brackets).

    Args:
        address: Such as ``127.0.0.1:8200`` or ``[::1]:8200``.

    Returns:
        The host, brackets removed.
    """
    host, _, _ = address.rpartition(":")
    return (host or address).strip("[]")


def is_loopback(address: str) -> bool:
    """Say whether an address is this machine's loopback interface.

    Args:
        address: ``host:port``.

    Returns:
        True for any loopback address (127.0.0.0/8, ::1, an IPv4-mapped one)
        and for ``localhost``.
    """
    host = host_of(address)
    if host == LOCALHOST:
        return True
    try:
        found = ipaddress.ip_address(host)
    except ValueError:  # a name lsof could not turn into an address
        return False
    mapped = getattr(found, "ipv4_mapped", None)
    return found.is_loopback or bool(mapped and mapped.is_loopback)


def watched_pids(root: int, table: str) -> list[int]:
    """Find a process, every process below it, and every one in its group.

    The harness starts in a session of its own, so its process id is also
    its group's id; a background command whose parent exited is adopted by
    process 1, but stays in that group.

    Args:
        root: The harness's process id (and process group id).
        table: ``ps -A -o pid=,ppid=,pgid=``'s output.

    Returns:
        Their ids, the root first.
    """
    children: dict[int, list[int]] = defaultdict(list)
    grouped = []
    for line in table.splitlines():
        parts = line.split()
        if len(parts) != PS_FIELDS or not all(part.isdigit() for part in parts):
            continue
        pid, parent, group = map(int, parts)
        children[parent].append(pid)
        if group == root:
            grouped.append(pid)
    found: list[int] = []
    seen: set[int] = set()
    queue = deque([root, *grouped])
    while queue:
        pid = queue.popleft()
        if pid not in seen:
            seen.add(pid)
            found.append(pid)
            queue.extend(children[pid])
    return found


def parse_lsof(output: str) -> list[tuple[str, str]]:
    """Read ``lsof -F pcn``'s output: one entry per socket.

    Args:
        output: Lines starting with ``p`` (process id), ``c`` (its command),
            ``f`` (a file number) and ``n`` (the socket's addresses).

    Returns:
        ``(command, addresses)`` pairs.
    """
    command, sockets = "", []
    for line in output.splitlines():
        if line.startswith("c"):
            command = line[1:]
        elif line.startswith("n"):
            sockets.append((command, line[1:]))
    return sockets


class NetworkWatch:
    """Sample a process tree's internet sockets until told to stop."""

    def __init__(self, root: int, interval_s: float = INTERVAL_S) -> None:
        """Remember what to watch.

        Args:
            root: The harness's process id.
            interval_s: Seconds between samples.
        """
        self.root = root
        self.interval_s = interval_s
        self.report = NetworkReport()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.started = 0.0

    def start(self) -> None:
        """Start sampling in the background."""
        self.started = time.monotonic()
        self.thread.start()

    def stop(self) -> NetworkReport:
        """Stop sampling (the sample under way, if any, is finished first).

        Returns:
            What the samples saw.
        """
        self.done.set()
        if self.thread.ident is not None:  # it was started
            self.thread.join()
        return self.report

    def loop(self) -> None:
        """Sample, then wait, until stopped."""
        while True:
            self.sample()
            if self.done.wait(self.interval_s):
                return

    def sample(self) -> None:
        """List the watched processes' sockets once and add them to the report.

        A sample that cannot be taken (``ps`` fails, or either command cannot
        run or overruns its time) is counted in ``failed`` instead.
        """
        try:
            listing = subprocess.run(
                [PS, "-A", "-o", "pid=,ppid=,pgid="],
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_S,
                check=False,
            )
            if listing.returncode != 0:
                self.report.failed += 1
                return
            pids = ",".join(str(pid) for pid in watched_pids(self.root, listing.stdout))
            # lsof exits 1 when it finds no socket: an answer, not a failure.
            output = subprocess.run(
                [LSOF, "-nP", "-a", "-i", "-p", pids, "-F", "pcn"],
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_S,
                check=False,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            self.report.failed += 1
            return
        self.report.samples += 1
        self.report.seconds = time.monotonic() - self.started
        for command, addresses in parse_lsof(output):
            if ARROW in addresses:
                remote = addresses.split(ARROW, 1)[1]
                self.report.connections.setdefault(remote, set()).add(command)
            else:
                self.report.unconnected.setdefault(addresses, set()).add(command)


def describe(report: NetworkReport, proxy: str) -> str:
    """Say in one line what the watch saw.

    Args:
        report: The samples.
        proxy: The proxy's ``host:port``, named when seen.

    Returns:
        The line.
    """
    head = (
        f"lsof every {INTERVAL_S} s: {report.samples} samples over "
        f"{report.seconds:.1f} s"
    )
    if report.failed or not report.samples:
        outside = f"; seen off this machine: {', '.join(report.outside)}"
        lost = f"{head} · WATCH FAILED: {report.failed} samples could not be taken"
        return lost + (outside if report.outside else "")
    if not report.connections:
        return f"{head} · no connection open at any sample"
    seen = ", ".join(
        f"{remote}{' (the proxy)' if remote == proxy else ''}"
        for remote in sorted(report.connections)
    )
    if report.loopback_only:
        return f"{head} · connections to {seen} only: loopback, nothing else"
    return f"{head} · NOT loopback only: {', '.join(report.outside)} (all: {seen})"
