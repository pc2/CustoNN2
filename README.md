## Getting Started

### Command line instructions
On the cluster infrastructure of PC2, you will be using the command line interface to manage your git repository.

Setup your git settings (since your home directory is mounted from every machine in the PC2 clusters, you only have to do this once).

```
git config --global user.name "<Firstname Lastname>"
git config --global user.email "<your email address>"
```

Clone the repository

```
git clone git@git.uni-paderborn.de:cs-hit/pg-custonn2-2018.git
```

### Trouble shooting

Some combination of ssh keys and the **xrdp** desktop connection lead to problems accessing the remote repository. The corresponding error message contains 

**Agent admitted failure to sign using the key.**

The following fix only applies if the above text is part of the error message. The fix for this problem involves setting the environment variable SSH\_AUTH\_SOCK=0. You can either prepend this statement to every git command that involves the remote, for example:

```
SSH_AUTH_SOCK=0 git clone git@git.uni-paderborn.de:cs-hit/pg-custonn2-2018.git
SSH_AUTH_SOCK=0 git pull
SSH_AUTH_SOCK=0 git push
```

Or you can permanently export this variable, for example from the .bashrc file in your home directory, e.g.

```
gedit ~/.bashrc &
```

and adding the line

```
export SSH_AUTH_SOCK=0
```

## Tutorial tasks

- 17.10.2018: [Lab setup, documentation, workflow](workflows/lab_instructions.md)
    - Due to 23.10.2018
    - [Individual Findings on workflows, to be kept up to date](workflows)
    - [Summary of findings from documentation](workflows/lab_results.md)
- 23.10.2018: [Programming Task 1](tutorial/Task1Instructions/Task1.md)
    - Exercises 1+2 due to 29.10.2018 - Submission by Email
    - **Exercise 3 due to 05.11.2018 - Submission by Email**
- 29./30.10.2018: [Presence Exercise](tutorial/ExampleKernels/README.md)
- 30.10.2018: [Mixed Practical and Research Task 2](tutorial/Task2/README.md)
    - **Due to 05.11.2018 - Submission through gitlab and with handwritten or digital/printed notes on the CNN lectures.**