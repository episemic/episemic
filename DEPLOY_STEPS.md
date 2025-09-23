### Depolyment steps

The deployment will first happen on the test PyPi first, and then will be pushed to production pypi.

#### Test PyPi

1. Run the tests, and make sure all tests are passing.
2. Once we are passing tests, lets do bumpversion.
3. Once bumpversion we commit and push to the main branch.
4. Then we use gh cli to create a release tag for the version.
5. Create the release on github for that version tag.
6. Then we push the code to test pypi for deployment of the library.

#### Production PyPi

1. From the test pypi release we will proceed with just pushing the version to prod.
