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

#### Eclipse projects + git

See [these instructions.](workflows/eclipse+git.md)

#### xrdp + git

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

#### Pulling or rebasing to master created merge conflicts

- If the merge conflicts are in actual source files, you need to manually resolve them and commit the merged files. Never commit files with the `>>>>>>>>>>>`, `==========`, `<<<<<<<<<<` conflict markers.
- If the merge conflicts are in eclipse .metadata files, you will just keep the state from the repository (functioning workspace). In that case you can use `git checkout --theirs .metadata`. (See also [this reference](http://gitready.com/advanced/2009/02/25/keep-either-file-in-merge-conflicts.html).)

## Tutorial tasks

- 17.10.2018: [Lab setup, documentation, workflow](workflows/lab_instructions.md)
    - Due to 23.10.2018
    - [Individual Findings on workflows, to be kept up to date](workflows)
    - [Summary of findings from documentation](workflows/lab_results.md)
- 23.10.2018: [Programming Task 1](tutorial/Task1Instructions/Task1.md)
    - Exercises 1+2 due to 29.10.2018 - Submission by Email
    - Exercise 3 due to 05.11.2018 - Submission by Email. I have seen some solutions that have not been submitted by Email, please still submit!
- 29./30.10.2018: [Presence Exercise](tutorial/ExampleKernels/README.md)
- 30.10.2018: [Mixed Practical and Research Task 2](tutorial/Task2/README.md)
    - Due to 05.11.2018 - Submission through gitlab and with handwritten or digital/printed notes on the CNN lectures.
    - Completion of all parts until 12.11.2018.
- 05.11.2018: [Practical Group Task 3: Implementing Linear classifier for MNIST on FPGA](tutorial/Task3/README.md)
	- Have functioning software and OpenCL kernel until 12.11.2018.
	- Complete task by 19.11.2018.
	- Complete hardware measurements, prepare presentation by 26.11.2018.
	- **More detailed analysis, hardware profiling, performance model until 14.01.2018**

- 09.11.2018: [Documentation, Tooling and Research Task 4](tutorial/Task4/README.md)
	- Watching and taking notes of Stanford CNN lectures until 12.11.2018.
	- Completion of all all other parts until 15.11.2018.
	- **Exentension of documentation until 03.12.2018**

- 13.11.2018:  [Applying Optimization Techniques Task 5](tutorial/Task5/README.md)
    - Completion until 23.11.2018.
	- **Submission of missing solutions via git, create a new merge request if the previous one was rejected.**
	
- 13.11.2018: Watching and taking notes of Stanford CNN lectures 5+6
    - Completion until 19.11.2018.
    - Prepare for a small test. You may use your notes and printouts, but no laptops or phones.

- 19.11.2018: [Practical implementation with Channels Task 6](tutorial/Task6/README.md)
    - **Completion until 29.11.2018.**

- 26.11.2018: [Practical Group Task 7: Implementing a small CNN on FPGA](tutorial/Task7/README.md)
    - **Completion of software reference until 03.12.2018**
    - **Completion of OpenCL kernels and working hardware design until 10.12.2018**
    - **Completion of optimized hardware and performance models until 14.01.2019**
    - **Exploration of design alternatives with higher accuracy until 14.01.2019**
 
- 26.11.2018: Watching and taking notes of Stanford CNN lectures 7-9
    - **Completion until 10.12.2018**

- background: Organize further group work
    - Work during term break, coordination of vacations.
    - Collect ideas for a project plan.