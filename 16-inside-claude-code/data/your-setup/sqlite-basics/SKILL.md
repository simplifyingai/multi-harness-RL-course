---
name: sqlite-basics
description: How to explore a SQLite database from the shell. Use when a task mentions a .db file.
---

# Exploring a SQLite database

1. List the tables: `sqlite3 data.db .tables`.
2. Read one table's columns: `sqlite3 data.db ".schema patients"`.
3. Ask one question per query: `sqlite3 data.db "SELECT COUNT(*) FROM patients"`.
