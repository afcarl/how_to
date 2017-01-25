## How to write python unit tests

<pre><code>import unittest

class YourTests(unittest.TestCase):
  def setUp(self):
    # where you do the setups

  def tearDown(self):
    # where you do the cleaning

  def test1(self):
    # test 1
    # ...

  def test2(self):
    # test 2
    # ...

if __name__ == '__main__':
  unittest.main()
</code></pre>

Auto-discover unit tests:

`python -m unittest discover`

## Show test coverage with Nose and Coverage

Install coverage:

`pip install coverage`

Use coverage:

`coverage run <testfilename>`
`coverage report -m`

Install nose:

`pip install nose`

Automatic discover unit tests and show test coverage:

`nosetests --with-coverage`

Output only the coverage of the files in current directory:

`nosetests --with-coverage --cover-inclusive --cover-package=.`
