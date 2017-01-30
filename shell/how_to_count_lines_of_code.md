## How to Count Lines of Code Under Directory

`find . -name '*.[postfix]' | xargs wc -l`

- Example:

`find . -name '*.py' | xargs wc -l`