# Setting up the Environment

Before you can run code for this tutorial, you need to set up your environment.

This section walks you through:
1. Setting up your Python environment.
2. Verifying everything is working.
3. Getting the tutorial code.

## Step 1: Set Up Your Python Environment

We provide a containerized environment that includes all the necessary
dependencies for this tutorial. To access this environment, follow the
instructions given
[here](https://github.com/TRISEP-2025-ML-tutorials/Intro-notebooks/blob/main/SETTING_UP.md).


## Step 2: Verify the Setup

To verify that everything is set up correctly, run the following command:

```bash
python -c "import torch; print(torch.__version__)"
```

If you see the version of PyTorch printed without any errors, your setup is
successful.

## Step 3: Get the Tutorial Code

First, fork
[this repository](https://github.com/TRISEP-2025-ML-tutorials/AdvancedTutorial)
to your own GitHub account. This allows you to make changes and save your
work without affecting the original repository. You can find instructions on how
to fork a repository
[here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository).

Finally, clone your forked repository from within a terminal session inside the
container:

```bash
git clone https://github.com/YOUR-USERNAME/AdvancedTutorial.git
```
