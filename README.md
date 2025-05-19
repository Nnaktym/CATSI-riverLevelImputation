# addr-development-template

This is the development template repository for ADDR.

## Usage

The usage instructions are documented in [Notion](https://www.notion.so/specteedev/Git-Rules-af412300324d4aae8fd2f142d45ea038?pvs=4).

### How to use this template

Please open the [repository URL](https://github.com/Spectee/addr-development-template)

If you choose `Use this template -> Create a new repository`

You can create a new repository that is a copy of this template repository."

> [!WARNING]
> Workflow secrets may not be moved when using this repository as template.
> If the secret information has not been transferred, please add it manually.

## Directory structure

```bash
.
├── .github # CI/CD Pipeline
│    └── workflows
│       ├── cicd-pipeline.yml
|       ├── dev-pipeline.yml
│       ├── deploy.yml
│       ├── slack-notification.yml
│       ├── static-analysis.yml
│       └── unit-tests.yml
├── .vscode # VsCode settings
│    ├── settings.json 
│    └── extensions.json
├── .prettierrc # coding rules for JS, TS, JSON
├── pyproject.toml # coding rules for ruff and mypy
└── requirements-lint.txt # install ruff mypy and stub libs for CI/CD Pipeline static-analysis.yaml
```

## VsCode settings

Please ensure that `.vscode` directory is always present in the workspace when developing.

The settings are described in the [settings.json](./.vscode/settings.json).

### Extension

The recommended and deprecated extensions are documented in [extensions.json](./.vscode/extensions.json)"

#### Recommendations

This is a list of installed extensions.

| Name          | ID                                           | Language  | Note               |
| ------------- | -------------------------------------------- | --------- |------------------- |
| Python        | `ms-python.python`                           | Python    | language support   |
| Pylance       | `ms-python.vscode-pylance`                   | Python    | language support   |
| Ruff          | `charliermarsh.ruff`                         | Python    | Formatter, Linter  |
| Mypy          | `ms-python.mypy-type-checker`                | Python    | Type checker       |
| autoDocstring | `njpwerner.autodocstring`                    | Python    | Docstring generator |
| markdownlint  | `davidanson.vscode-markdownlint`             | Markdown  | Formatter, Linter  |
| Prettier      | `esbenp.prettier-vscode`                     | JS, TS, JSON | Formatter       |
| Code Spell Checker | `streetsidesoftware.code-spell-checker` | Any          | Spell checker   |

#### Unwanted recommendations

This is a deprecated extension. If already installed, please either disable or uninstall it.

| Name          | ID                             | Language  | Note                         |
| ------------- | ------------------------------ | --------- |----------------------------- |
| Black        | `ms-python.black-formatter`     | Python    | Formatter is done with Ruff  |
| Flake8       | `ms-python.flake8`              | Python    | Linter is done with Ruff     |

## Python settings

### Formatter Linter

Formatter and Linter is [Ruff](https://docs.astral.sh/ruff/).

The configuration for Ruff is specified in [pyproject.toml](./pyproject.toml).

Please add the settings you want to disable for each repository to the `ignore` in the configuration.

### Type checker

Type checker is [mypy](https://mypy.readthedocs.io/en/stable/)

The configuration for mypy is specified in [pyproject.toml](./pyproject.toml).

Please add the settings you want to disable for each repository to the `tool.mypy` in the configuration.

## TypeScript settings

### Formatter

Formatter is [Prettier](https://prettier.io/).

The configuration for prettier is specified in [.prettierrc](./.prettierrc).

## CI/CD Pipeline

The template for conducting CI/CD using GitHub Actions is located in `.github/workflows`.

- [cicd-pipeline.yml](./.github/workflows/cicd-pipeline.yml)
- [dev-pipeline.yml](./.github/workflows/dev-pipeline.yml)
- [static-analysis.yml](./.github/workflows/static-analysis.yml)
- [unit-tests.yml](./.github/workflows/unit-tests.yml)
- [deploy.yml](./.github/workflows/deploy.yml)
- [slack-notification.yml](./.github/workflows/slack-notification.yml)

### Settings

#### cicd-pipeline.yml

This serves as the entry point to trigger CI/CD.

Please customize the `on: push` to meet specific conditions.

For example, if you want to trigger it only when pushing to the `main` branch or the `story0` branch, you can write it as follows.

```yaml
on:
  push:
    branches:
      - main
      - story0
```

#### dev-pipeline.yml

Any workflow in the dev pipeline has to be activated manually. It uses the same scripts as the cicd pipeline, but each section can be run individually.

#### static-analysis.yml

This file performs the following actions.

- Static code analysis using ruff and mypy
- Spell check using cspell

Please add the stub packages used for static analysis to [requirements-lint.txt](./requirements-lint.txt).
Example, add the stub package for `requests` called `types-requests`.

```txt
# this is a lint package list for cicd pipeline
# linter packages. (mypy and ruff)
mypy>=1.7.1
ruff>=0.1.2
# Specify the stub packages that are required for mypy to type check the dependencies.
types-requests==2.31.0.10
```

To initiate this flow, the following arguments are required.

Align the Python version with the one used in development (containers, Lambda, etc.)

```yaml
lint_dir:
  description: "Path to the directory to lint"
  required: true
  type: string
python_version:
  description: "Python version to use"
  required: true
  type: string
```

Example how to specify it in `cicd-pipeline.yml`.

```yaml
your_static_analysis:
  uses: ./.github/workflows/static-analysis.yml
  with:
    lint_dir: "package/your_handler_name/src/"
    python_version: "3.11"
```

#### unit-tests.yml

This file performs the following actions.

- We will execute tests using pytest.

After running the tests, we will generate a coverage report.

To initiate this flow, the following arguments are required.

Align the Python version with the one used in development (containers, Lambda, etc.)

```yaml
package_dir:
  description: "Path to the package directory containing the module to be tested"
  required: true
  type: string
test_dir:
  description: "Path to the test directory"
  required: true
  type: string
requirements_file:
  description: "Path to the requirements file containing the dependencies"
  required: true
  type: string
python_version:
  description: "Python version to use"
  required: true
  type: string
```

Example how to specify it in `cicd-pipeline.yml`.

```yaml
your_unit_test:
  # Please add the condition for the static analysis job to the 'needs' section.
  needs: [your_static_analysis]
  uses: ./.github/workflows/unit-tests.yml
  with:
    package_dir: "package/your_handler_name/src/"
    test_dir: "tests/"
    requirements_file: "package/your_handler_name/requirements.txt"
    python_version: "3.11"
```

#### deploy.yml

This file performs the following actions.

- We will execute cdk deploy.

To initiate this flow, the following arguments are required.

```yaml
secrets:
  AWS_ROLE_ARN_DEV:
    required: true
  AWS_ROLE_ARN_PROD:
    required: true
```

Example how to specify it in `cicd-pipeline.yml`.

```yaml
your_deploy:
  needs: [your_unit_test, your_static_analysis]
  uses: ./.github/workflows/deploy.yml
  secrets:
    AWS_ROLE_ARN_PROD: ${{ secrets.AWS_ROLE_ARN_PROD }}
    AWS_ROLE_ARN_DEV: ${{ secrets.AWS_ROLE_ARN_DEV }}
```

##### AWS_ROLE_ARN_${ENV}

`${ENV} = DEV or PROD`

Retrieve the ARN of the deployment role (`githubaction`) from AWS IAM.

Go to the repository settings, choose [Actions secrets and variables](https://github.com/Spectee/addr-development-template/settings/secrets/actions).
and add a secret variable named `AWS_ROLE_ARN_${ENV}` to the Repository secrets.

#### slack-notification.yml

This file performs the following actions.

- When the workflow fails, send a notification to Slack.

The channel to notify is `engineer_adv_drr_github_notice`.

To initiate this flow, the following arguments are required.

```yaml
secrets:
  ADDR_CICD_FAILURE_SLACK_WEBHOOK_URL:
    required: true
```

Example how to specify it in `cicd-pipeline.yml`.

```yaml
slack_notification:
  # please add the condition for the [static_analysis, unit_test, deploy] job to the 'needs' section.
  needs: [your_static_analysis, your_unit_test, your_deploy]
  if: ${{ failure() }}
  uses: ./.github/workflows/slack-notification.yml
  secrets:
    ADDR_CICD_FAILURE_SLACK_WEBHOOK_URL: ${{ secrets.ADDR_CICD_FAILURE_SLACK_WEBHOOK_URL }}
```

##### ADDR_CICD_FAILURE_SLACK_WEBHOOK_URL

`ADDR_CICD_FAILURE_SLACK_WEBHOOK_URL` has already been configured in the [organizational environment variables](https://github.com/Spectee/addr-development-template/settings/secrets/actions).
