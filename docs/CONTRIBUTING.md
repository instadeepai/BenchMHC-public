# Contributing

!!! info

    The following is a set of guidelines for contributing to the `BenchMHC-public` repo.

## How to create an issue

The GitHub issues are used to track all our tasks (e.g. feature, bug fix, ...).

1. Go to the [issues list](https://github.com/instadeepai/BenchMHC-public/issues).
2. Click on "New issue" in the top right corner.
3. Choose an issue template.
4. Fill in the relevant details for the issue.
5. (Optional) Assign yourself to the issue by clicking on "Assign yourself".
6. (Optional) The relevant label will be automatically added based on the template you use but
   you can choose to select any other relevant labels.
7. Click on "Submit new issue" in the bottom of the page.

## How to contribute

### For maintainers (write access to the repository)

You implement your changes locally, in a new branch:

1. Assign yourself to an issue.
2. Click on "Create a branch" on the right of the issue page: it will create a branch with the
   pattern `{issue_id}-{issue_title_slug}`. The pull request opened for this branch will be
   automatically associated to the issue.
3. Locally, fetch and checkout the branch:

   ```bash
   git fetch
   git checkout {branch} # use autocomplete by writing {issue_id}- and then press TAB
   ```

4. Implement your changes, commit and push.

   :warning: your commit messages must follow our
   [git conventions](CONTRIBUTING.md#git-conventions).

Once you are done, you ask for a review of your code before merging it into
the `main` branch:

1. Go to the [branches list](https://github.com/instadeepai/BenchMHC-public/branches).
2. Select the branch that you want to merge and click on "Create a PR".
3. Fill in the pull request title: use the title of your main commit.
4. Create the pull request. If not ready yet, you can also create a draft pull request by clicking
   on "Create Draft PR".
5. Select an appropriate label and assign yourself to the PR. :warning: Labels are important for
   meaningful release notes.
6. Once your code is ready, set the PR as ready by clicking on "Ready for review" on the bottom of
   the page. Make sure to tick the checkbox in the PR description to ensure you follow contributing
   guidelines.
7. Assign a reviewer.
8. Integrate the feedbacks if any:
   - The author of the conversations (most often the reviewer) on a PR is in charge of marking
     them as resolved in GitHub UI.
   - To answer the reviewer's comments in one batch instead of one by one, go to the "Files
     Changed" tab (or click on "Review now" on the upper right side), scroll down until you find
     the comments (or you can use the "Conversations" dropdown to select a comment), then click
     on "Start a review" on your first reply, and then on the "Add to review" button on the next
     ones. Once completed, you can click on the "Review changes" dropdown in the upper right
     corner and click on "Comment", this will finish the review.
   - Once done, re-request a review by clicking on the round arrow in the "Reviewer" section.
9. Once your PR is approved:
   - Update your branch with the latest version of `main`:

   ```bash
   git fetch && git rebase origin/main && git push -f
   ```

   - Clean your commits if necessary: you should keep one commit per task (e.g. if in your
   branch you did a refactoring and a new feature you should have 2 commits).
   - Merge! :tada:

### For external contributors (fork-based workflow)

If you do not have write access to the repository, you can contribute via a fork:

1. Open an issue first. Before starting work, open an issue describing the bug or feature.
   This lets us discuss the approach before you invest time.
2. Fork the repository by clicking "Fork" on the
   [GitHub page](https://github.com/instadeepai/BenchMHC-public).
3. Clone your fork and create a branch:

   ```bash
   git clone https://github.com/<your-username>/BenchMHC-public.git
   cd BenchMHC-public
   git checkout -b <issue_id>-<short-description>
   ```

4. Implement your changes, following the commit conventions described in
   [Git conventions](CONTRIBUTING.md#git-conventions).
5. Push to your fork and open a pull request against `instadeepai/BenchMHC-public:main`:

   ```bash
   git push origin <your-branch>
   ```

   Then click "Compare & pull request" on GitHub.
6. Fill in the PR template, select an appropriate label, and reference the issue
   (e.g. `Closes #123`).
7. Wait for the CI to succeed and ask for a review. See 8. and 9. from the [maintainers
   section](#for-maintainers-write-access-to-the-repository) for more details.

!!! note "CI on fork pull requests"

    Fork PRs run the full CI pipeline (linting, tests, and documentation build) but without
    Docker registry access. Images are built locally on the runner instead of being cached in
    GHCR. This means fork CI builds may be slower than internal ones. The coverage PR comment
    is also skipped on fork PRs.

## Pre-commit hooks

To ensure code quality we are using [pre-commit hooks](https://pre-commit.com/), make sure you
have installed it (cf. the [Pre-commit hooks](guides/development/pre_commit_hooks.md) guide)!

## Git conventions

### Commit message format

- The section relies on the
  [Contributing to Angular - Commit Message Guidelines](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit).
- It provides conventions to write commits messages based on the
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.2/).
- It aims to :
  - Get a well-structured and easily understandable git history.
  - Generate changelogs easily for each release since we can use scripts
    that parse the commit messages.
- The commit messages must have the following structure :

```markdown
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

- `<type>` section :
  - It is mandatory.
  - It must be one of the following :
    - `build`: Changes to our deployment configuration (e.g. docker,
      requirements, pre-commit configuration).
    - `ci` : Changes to our CI configuration files and scripts.
    - `docs` : Documentation or configuration file changes.
    - `feat` : A new feature.
    - `fix` : A bug fix.
    - `perf` : A code change that improves performance.
    - `refactor` : A code change that neither fixes a bug nor adds a feature.
    - `style` : Changes that do not affect the meaning of the code (white-space,
      formatting, missing semicolons, etc.).
    - `test` : Adding missing tests or correcting existing tests.
- `<scope>` section :
  - It is mandatory except for `EXP` commits.
  - It describes the module affected by the changes.
  - Conventions for `<scope>` section:
    - If you want to add a new module in the repo (ex: `bench_mhc/cli/my_new_command.py`) you
      can use the name of the parent directory as the scope (ex: `feat(cli): add
      <my_new_command_line> command line`) since the module (`my_new_command_line.py`) does not
      yet exist before the commit.
    - If you just modified an already existing module  (ex: `my_new_command_line.py`), you can
      directly use the file name in the scope (ex: `feat(my_new_command_line): add my new feature`).
    - If your changes affect the whole repo, use `all` for the scope `style(all): change mhc
      naming convention`.
    - If your changes affect several modules, you can use comma separated scope (ex: `feat
      (calibrate,train): add new training method`).
- `<subject>` section :
  - It is mandatory.
  - It contains a succinct description of the change.
  - Few recommendations about the subject :
    - Use the imperative, present tense: "change" not "changed" nor "changes".
    - Don't capitalize the first letter.
    - No dot (.) at the end.
- `<body>` section :
  - It is optional.
  - It is an extension of the `<subject>` section used to add a longer description about the
    changes if relevant.
  - If you use the `<body>` section to list other changes in the commit, format these changes
    using `- <type>(<scope>): <subject>` to be aligned with the format of the main commit message,
    e.g. like in the example commit message below.
- `<footer>` section :
  - It is optional.
  - It can contain information about breaking changes and is also the place to reference GitHub
    issues, that this commit closes or is related to.

Example commit message:

```markdown
feat(cli): add <my_new_command_line> command line

- fix(cli): make <this_other_command_line> command line work with this parameter
- style(all): change mhc naming convention

- BREAKING CHANGE: `extends` key in config file is now used for extending other config files
```

- You can add the commit message template to the git configuration by running :

```bash
git config commit.template $PWD/.gitmessage
```

### Pull request rules

- The commit history shall be as atomic as possible: one commit per task.
- For very simple and trivial modifications (e.g. typo correction, very light refactor), it is
  not necessary to create a dedicated PR, but simply possible to integrate them into one of your
  current PR.
- The author or reviewer may prefer a dedicated PR if the modification is already (or becomes)
  too large and requires a proper review.

### Git checklists

#### For authors 🧑‍💻 (before marking your PR as ready)

- Check that **the commit history is clean**, i.e. explicit and comprehensive; and that the
  modifications of your commits are consistent and atomic.
- Check that all your commits **respect the present guidelines**, notably
  [commit message format](CONTRIBUTING.md#commit-message-format).

#### For reviewers 🕵️ (before approval)

- Check changes: do not hesitate to give feedback for any questions or concerns.
  **Feedback is good for everyone, authors and reviewers!**.
- Check that all commits **respect the present guidelines**. In particular, check that the
  history does not contain superfluous commits; request to have these commits removed if necessary.

#### For both 🧑‍💻 🕵️ (during the review)

- Make your suggestions/comments in one batch instead of one by one. For that, you need to click
  on the "Start a review" button on your first comment and then on the "Add to review" button
  on the next ones. Once completed, you can click on "Finish review".

## Style Guide

Here are guidelines regarding some aspects that couldn't be checked by the
`pre-commit`.

### Docstring and comments

- For the docstring, follow the
  [Google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Docstrings are mandatory for all classes / functions / methods / modules.
- To note that in our case we set the maximum character length of our docstring to **100**. This
  is because we are using Ruff and its configuration is set to wrap after 100 characters length
  of code. It can be useful to configure a vertical line at 100 characters in your IDE (e.g.
  `Hard Wrap` in PyCharm).
- About the docstring structure:
  - For `Returns`, we should add the description of the output on a new line.
  - For `Raises`, we should have 1 error per line with the following pattern
  `{ErrorClass}: {why}`. If the same error can be raised for multiple
  reasons, mention the error once and use bullet points, e.g.:

  ```python
  """
  Raises:
      {ErrorClass}:
          - {why 1}.
          - {why 2}.
  """
  ```

  - For multiline description of `Args`, we should indent the description.
  - For `Returns`, `Raises` and `Args` we should always capitalize the first letter and end with
    a period e.g. `This is an Args, Raises or Returns docstring.`.
  - The docstring should start with a verb in its infinite form. You should use an imperative
    style (“Do this”, “Return that”) not a description style ("Returns the pathname …”). Even
    though it is not mandatory in the Google style guide, we use it as a convention.
- Do not specify typing in docstring.
- For links to papers, we should use the non-PDF link (e.g.
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6885703/> instead of
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6885703/pdf/zjw2459.pdf>).
- When dealing with torch tensors or numpy arrays, put the shape as comment in the line before
  (or in the docstring description for function's input/output) and use `?` for batch size, e.g.:

  ```python
  peptide = self.inputs["peptide"]
  peptide_encoder = self._build_input_encoder(peptide)

  # (?, peptide_len)
  peptide_input = peptide.build_input_layer()

  # (?, peptide_len, d_model)
  peptide_encoded = peptide_encoder(peptide_input)
  ```

- Use comments only if the code is not self-explanatory. Comments should also start with a
  capitalized letter and end with a period e.g. `This is a comment.`.
- For command lines only:
  - Add blank lines between bullet points in docstrings and two horizontal spaces.
  - Include in the docstring a link to the documentation that refers to this command line.

### Function structure

- Add blank line before the `return` statement in a function, except:

- When the function body is only the `return` statement, e.g.:

  ```python
  def my_func(param: int) -> int:
    """Super docstring."""
    return 2 * param
  ```

- If the `return` statement is after a logical statement, e.g.:

  ```python
  def my_func(param: int) -> int:
    """Super docstring."""
    if param % 2 == 0:
      return param + 4

    return 2 * param
  ```

- Do not add blank line after `for`/`if`/`elif`/`else`/`while` statements.
- Use blank lines in functions to separate logical blocks.
- Do not add a blank line between `try`/`except`/`else`/`finally` statements, e.g.:

  ```python
  try:
    result = x // y
    print(f"Your answer is : {result}")
  except ZeroDivisionError:
    print("You are dividing by zero")
  ```

- For the typing of a generator which only yield values (i.e. no definition of `send` or `return`),
  you should use `collections.abc.Iterator[YieldType]` instead of `collections.abc.Generator
  [YieldType, None, None]`.

### Data wrangling

- To assign a new column with `polars.{DataFrame.LazyFrame}.with_columns` prefer using `{polars.
  Expr}.alias("my_column_name")` over `my_column_name={polars.Expr}` (not working when column
  name is stored in a variable).
- Name pl.DataFrame objects as `df` or `df_{misc}`.
- Name pl.LazyFrame objects as `lf` or `lf_{misc}`.
- Prefer the lazy API (`polars.LazyFrame`) for multi-step pipelines so polars can optimize the
  query plan; collect only when you need materialized data.

### Miscellaneous

- Use `f-string` to format variables in string instead of `%` or `format`.
- Use dot-ended sentences in logging statements e.g. `log.info("Doing this.")`.
- Use list comprehension instead of `map`/`filter`.
- Use `pathlib.Path` to deal with local paths.

  ```python
  from pathlib import Path

  import polars as pl

  def load_dataset(file_path: Path) -> pl.DataFrame:
    ...

  my_file_path = Path("..")
  ```

- If you need a container for membership test, e.g. `if "test" in {"test", "another_test"}: print
  ("Found!")`, use a `set` as advised by [`pylint`](https://pylint.pycqa.org/en/latest/user_guide/messages/refactor/use-set-for-membership.html). <!-- markdownlint-disable MD013 -->
- If you need a container for iteration, e.g. `for element in ("test", "TEST",
  4)`, prefer using a `tuple` over a `list`, a `frozenset`, or a `set`,
  since it is advised by [`pylint`](https://pylint.pycqa.org/en/latest/user_guide/messages/convention/use-sequence-for-iteration.html) <!-- markdownlint-disable MD013 -->
  to use `list`/`tuple`/`range` for iteration and since `tuple` is faster to be created (see
  benchmark below). If you consider iterating over consecutive numbers though, use `range` as it
  is more convenient/readable.

  ```python
  def from_list():
      return ["test", "TEST", 4]

  def from_tuple():
      return ("test", "TEST", 4)

  def from_set():
      return {"test", "TEST", 4}

  def from_frozenset():
      return frozenset(("test", "TEST", 4))

  %timeit from_list() # 84.9 ns ± 1.44 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
  %timeit from_tuple() # 56.4 ns ± 0.687 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
  %timeit from_set() # 106 ns ± 3.99 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
  %timeit from_frozenset() # 177 ns ± 1.21 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
  ```

- When using an MkDocs'
  [admonition](https://squidfunk.github.io/mkdocs-material/reference/admonitions/) in the
  documentation, its type must be in lower case (e.g. `note`, `info`, `tip` or `example`).
- Use `_` as thousand separator to format large numbers with `f-string`, e.g.:

  ```python
  my_large_number = 262868846
  print(f"{my_large_number:_}") # 262_868_846
  ```

- Private functions should always be put after all public functions within a file.

### Naming conventions

- For MHC classes, use MHC1/mhc1 and MHC2/mhc2.
- For the evaluation metric, use _Top-K_ instead of _TopK_.
- For "number" abbreviation, use `num` instead of `nb`.
- When mentioning Google storage buckets, use `GS` instead of `GCP`.
- Camel-cased `BenchMHC` is used for the official name i.e. Github / MLflow / documentation while
  kebab-cased `bench-mhc` is used everywhere else, i.e. project name, PyPI package name, CI, etc.
  except snake-cased `bench_mhc` for the package folder / import name
  [to align with PEP8](https://peps.python.org/pep-0008/#package-and-module-names).
- For the names of the NetMHCpan-4.1 evaluation datasets, use _CD8 epitopes_ and _MS ligands_.
- For the names of the commands (and their associated modules), use an active verb instead of a noun, e.g. `train` / `train_command.py` / `train.py` instead of `training` / `training_command.py` / `training.py`.
