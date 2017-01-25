## How to improve python code style

#### Use [autopep8](https://pypi.python.org/pypi/autopep8)

- Installation

`$ pip install --upgrade autopep8`

- Usage

`$ autopep8 --in-place --aggressive --aggressive <filename>`

If you want to keep indentation style (Google uses two spaces for indentation):

`$ autopep8 --in-place --aggressive --aggressive --ignore=W191,E101,E111 <filename>`

#### Use [pylint](https://www.pylint.org/) to check python style

- Installation

`pip install pylint`

- Usage

`pylint <filename>`

Ignore warning:

`pylint --disable=W <filename>`