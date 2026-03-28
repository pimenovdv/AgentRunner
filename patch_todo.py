with open("todo.md", "r") as f:
    content = f.read()

content = content.replace(
    '- [ ] **5.2. Ограничение итераций (Loop Bounds):**',
    '- [x] **5.2. Ограничение итераций (Loop Bounds):**'
)

with open("todo.md", "w") as f:
    f.write(content)
