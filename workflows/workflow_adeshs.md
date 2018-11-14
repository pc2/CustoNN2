# CustoNN2

## ssh or xrdp for your normal workflow?
- I am okay with using either one of them.
- xrdp seems to be a little slow compared to ssh.
- For ssh, we need not use VPN when connecting from home.
- However for xrdp, we need VPN for accessing the Cluster when connecting from home.

## Cloned versions of the git repository
- One in the cluster in directory - /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs
- One more in the local machine.
- How to sync between repo : Using `git pull` command.

### Mount shared file system from the local system
Mounted the shared file system on my windows local machine using CIFS .
CIFS IMT Link : https://hilfe.uni-paderborn.de/Netzlaufwerk_einbinden_(Windows_7)

### Which documentation will you use frequently, how will you access it, do you need local copies of the relevant pdfs?
**TODO**

## TASK 2

### Which parts of the infrastructure do you expect to use within the project?
- Initially, we will be using the Custom Computing (CC) Cluster for development.
- After Development phase, we will be using HPC production Noctua Cluster.

### Which tools, FPGAs and boards do you expect to use within the project?
- We will be using **Nallatech 385A board with Intel/Altera Arria 10 GX 1150 FPGA** for development purpose.
- The Goal of this project is to deploy CNNs on **Nallatech 520N boards with Stratix 10 FPGA**.


## TASK 3

### Which of the sanity checks from the FPGA documentation can you perform this machine, does it contain FPGAs?
- `aoc -version` will display the compiler version.
- `quartus_cmd -version` will give us the version of the Quartus Prime.
- `aoc -list-boards` will list down the available boards connected to the machine.
- `aocl diagnose`  could not be executed since this check has to be done on FPGA Node.

### What is the path to your mounted user home on this system? What is  your quota here
- /upb/departments/pc2/users/a/adeshs
- Quota : 5GB

### Can you access and edit the .bashrc file there?
- Yes, .bashrc file is owned by me and I can edit it. We usually add environment variables inside .bashrc file. I added
`export PG_HOME_ADESH=/upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2/adeshs` as a variable for easy access of the project directory.
- Quota in Cluster is 15TB

## Channels & Pipes in OpenCL
- Channels/Pipes are OpenCL extensions or functions which provide data transfer functionality between kernels with low latency and high efficiency.
- FIFO mechanism communication and unidirectional
- Helps in in-order execution
- The default behavior of Channels is blocking whereas for pipe, it is non-blocking.
- Channels can have multiple read call but only a single write call.
- Pipes can have only a single read and a single write call.
- Channel implemented file scope variable
- Pipes are implemented as kernel arguments.
  
source:
- Intel FPGA SDK OpenCL Programming Guide : https://www.intel.com/content/www/us/en/programmable/documentation/mwh1391807965224.html
- Using Channels and Pipes with OpenCL on Intel FPGAs : https://www.youtube.com/watch?v=_0RtAKeRl00

## aoc useful commands
- -march = emulator Create kernels to be executed on x86
- -list-boards = prints all available boards
- -c -report = generates html report of the kernel for 17.1.2 tools
- -rtl -report = generates html report of the kernel for 18.0.1 tools

## Task 4 :Git Best Practices
  
- Enabling git color coded console:  
> git config --global color.ui auto
  
### Useful git commands:
- `git add <filenames>` moves the files from untracked state to  git staging area. These files will be included in the next commit.
- `git add .` adds all the untracked/modified files from the current directory.
- `git add -n .` This command doesn't actually adds the files, just shows which files will be added from the current dir.
- `git status` lists the status of the files in the git branch ( Untracked/modified/files to be committed )
- `get reset` tool for undoing the commits in a branch
- `git commit -m "<Message>"` Commits the newly added (tracked files) to the branch.
- `git push origin <branch_name>`  pushes all the commits from local repo branch to remote repository.
- `git pull` pulls all the changes from the remote repository to local repo.
- `git log` shows the commit logs 
### Git Rebase:
- Git Rebase helps us in combining/adding commits from one branch to another branch.
- It adds the new commits on top of the base commit. 
- If there are no conflicts, the new commits in another feature branch can be added to master branch using rebase.  
  
source:  
https://git-scm.com/docs

### Resolving git conflicts:
- Git Conflicts happens when I have commited a file locally and pull new changes from the remote repository and that file has competing commits.
- Git auto merges the file with conflict markers(<<<<< HEAD <local changes> ==== <Remote Changes> >>>>> commitID) , search for these conflict markers in the file.
  
- git pull log obtained in dev_branch_adeshs for conflict:
> Auto-merging workflows/workflow_adeshs.md
  CONFLICT (content): Merge conflict in workflows/workflow_adeshs.md
  
- this conflict was manually resolved and pushed back to the remote repository.
  
source:  
https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/