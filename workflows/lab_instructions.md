## Step 1
Setting up access to the git repository (make sure to start with the first two steps here!)

- In your browser, login once with your IMT login at https://git.uni-paderborn.de
- Ask me to add you as member of the PG repository ‘pg-custonn2-2018’. I can't do this before you have logged in once.

- After I have added you, login again at https://git.uni-paderborn.de, see that you now have access to https://git.uni-paderborn.de/cs-hit/pg-custonn2-2018
- If you are not familiar with git and gitlab, find out the essentials of the git workflow - feel free to use and share other resources
- Overview of gitlab
    - https://docs.gitlab.com/ee/user/
- Entry points to git
    - https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
    - https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository
    - https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell and following sections


## Step 2
Familiarize yourself with the FPGA / Custom Computing infrastructure and documentation at PC²

- In your browser, login with your IMT login at https://wiki.pc2.uni-paderborn.de/display/FPGAIn/Infrastructure+Overview
- Get an overview of the documentation provided here
- Which parts of the infrastructure do you expect to use within the project?
- Which tools, FPGAs and boards do you expect to use within the project?
- Read about the file system structure on those systems


## Step 3
Setup access to the CC Cluster

- Follow the wiki instructions to connect to the frontend of the CC Cluster
- Test both ssh connections and xrdp
- Which of the sanity checks from the FPGA documentation can you perform this machine, does it contain FPGAs?

- What is the path to your mounted user home on this system? What is your quota here (use `df -h` to find out)?
- Can you access and edit the .bashrc file there?
- In the file system, go to /upb/scratch/departments/pc2/groups/pc2-cc-user/custonn2 and setup a local working directory with your IMT username. What is your quota here (use again `df -h` to find out)?
- Once you have access to the git repository, clone a copy of this repository to this working directory


## Step 4
Planning and setting up your workflow, documenting it in git

- Do you prefer ssh or xrdp for your normal workflow? Be prepared to use both at times! How is the performance of each via cable in the lab, via eduroam, from your home?

- Where do you want to keep cloned versions of the git repository?
    - One copy in the cluster infrastructure - you will need that to use the OpenCL tools installed there
    - Another copy on your local machine for writing and editing documentation, etc? - If so, how will you synchronize between the two repositories?
    - Can and will you mount the shared file system from your local system? How is the performance of this via cable in the lab, via eduroam, from your home?

- Which documentation will you use frequently, how will you access it, do you need local copies of the relevant pdfs?

- Write a document about your personal findings and decisions in gitlab compatible markdown. Call this file workflow_<yourIMTLogin>.md, commit and push it to the project git repository and link it from the README.md file of the repository.
- Also summarize your findings from task 2 and 3 here.
- At this current stage, this can only contain first thoughts about your workflow, update the document over time. 