## Introduction to CNNs, Lectures 3 and 4

- Watch the next two lectures (3 and 4) of Stanford University cource CS231n Convolutional Neural Networks for Visual Recognition on youtube
	- https://www.youtube.com/watch?v=h7iBpEHGVNc&index=3&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
	- https://www.youtube.com/watch?v=d14TUNcbn1k&index=4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
- Take some notes on the overall course contents. What was particularly new or interesting to you?

## Documentation and Knowledge Base

**Make sure to find your own individual representation of topics here.**

- Extend your personal workflow.md file to a knowledge base:
	- Create a hierarchical list of topics that you have encountered during the tutorial phase. Topics on the lowest hierarchy level could be for example:
	 	- OpenCL channels
		- Gitlab merge requests
		- aoc command line options
		- Mounting university file systems
		- FPGAs and OpenCL SDK tool versions
	- For every topic, link to the sources (.pdf and websites) that have been used or should be kept in mind, including
		- Wiki
		- Gitlab
		- External websites

## Git best practices

Research about the following git topics, apply them practically and document your findings (including short code snippets) in your workflow.md file. 

**You can discuss and collaborate on background research here, but be sure to personally gain the practical experience of using git.**

- If your git console does not have colored output yet, enable color coded console output by changing your global git settings.
- Go through the following sequence of commands. What do the individual steps do, which of them can take a relative path argument or file argument (like `.`)? How can this sequence avoid problems when committing changes?

```
git status
git add -n
git add
git status
git reset
git status
git commit -m "Message"	
```

	
- Rebase your development branch on top of the master branch. What does it do and in which situations is this useful?
- See the outputs of `git log` and find out more details about individual commits in the command line and using the gitlab website.
- Using two different local checkouts of the repository and working in a development branch, create a conflict within your workflow.md file by performing and committing different changes in each of the repository copies. Push one of the commits, pull the changes from the other repository - here a conflict should arise - and resolve the conflict by a manual merge operation. Commit the merge and push to the remote repository again.