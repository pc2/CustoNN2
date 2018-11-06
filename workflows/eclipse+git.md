## Working with Eclipse projects and git

### Task1: Easy start, difficult synchronization

In Task1, you have started working with Eclipse. The directory `tutorial/Task1` contained the `.metadata` directory of an Eclipse workspace, which allowed you to directly start working with an existing project in the workspace. This was the goal of the original "ready-to-use" tutorial task. You can inspect the existence of the `.metadata` directory with `ll -a`.

```
cd pg-custonn2-2018/tutorial/Task1
[kenter@fe-1 Task1]$ ll -a
total 5507
drwxrwx---  5 kenter user     288 Nov  6 16:09 .
drwxrwx--- 14 kenter user     520 Nov  6 16:50 ..
-rw-rw-r--  1 kenter user    1004 Nov  6 16:09 .gitignore
-rw-rw-r--  1 kenter user     233 Oct 22 20:11 helpful_commands.txt
drwxrwx---  4 kenter user     180 Nov  6 16:18 .metadata
-rw-rw-r--  1 kenter user     509 Oct 22 18:29 opencl_init.sh
-rwxrwx---  1 kenter user     275 Nov  6 15:58 SimpleKernel.cl
drwxrwx---  4 kenter user     215 Nov  6 13:19 SimpleOpenCL
```

Unfortunately, the workspace `.metadata` does not only contain the used project, but also contains many more settings, log files, local file history and much more. Lacking a proper `.gitignore` file and special care during the `git add`, `git status` and `git commit` steps, changes to this workspace `.metadata` have been pushed to the repository, which seems to have rendered some projects unusable. I have cleaned up these commits and added a restrictive `.gitignore` file, so the current state of the master branch should be usable again.

In case of a conflict during `git pull`, caused by not commited local changes to any of your files in `.metadata`, you do not want to commit the changes, but rather revert to the original state using `git checkout .metadata` from the Task1 directory.

### Better alternative: Only share project settings

The more sustainable to share Eclipse projects through git is by only sharing the project files themselves (`.project` and `.cproject`), but keeping the workspace data `.metadata` completely out of the repository. I have commited a clean copy of the solution to Task 1 Exercise 2 to `pg-custonn2-2018/tutorial/Task1Exercise2SolutionCleanEclipseProject` that is setup this way. There is no `.metadata`, but the project settings are inside the SimpleOpenCL directory

```
cd pg-custonn2-2018/tutorial/Task1Exercise2SolutionCleanEclipseProject
[kenter@fe-1 Task1Exercise2SolutionCleanEclipseProject]$ ll -a
total 5505
drwxrwx---  4 kenter user     261 Nov  6 17:41 .
drwxrwx--- 14 kenter user     520 Nov  6 16:50 ..
-rw-rw-r--  1 kenter user    1124 Nov  6 16:36 .gitignore
-rw-rw-r--  1 kenter user     233 Nov  6 16:23 helpful_commands.txt
-rw-rw-r--  1 kenter user     509 Nov  6 16:23 opencl_init.sh
-rwxrwx---  1 kenter user     275 Nov  6 16:23 SimpleKernel.cl
drwxrwx---  4 kenter user     215 Nov  6 16:23 SimpleOpenCL
[kenter@fe-1 Task1Exercise2SolutionCleanEclipseProject]$ ll -a SimpleOpenCL/
total 281
drwxrwx--- 4 kenter user   215 Nov  6 16:23 .
drwxrwx--- 4 kenter user   261 Nov  6 17:41 ..
-rw-rw-r-- 1 kenter user 15115 Nov  6 16:23 .cproject
drwxrwx--- 2 kenter user   241 Nov  6 16:58 Debug
-rwxrwxr-x 1 kenter user  4667 Nov  6 16:23 main.cpp
-rw-rw-r-- 1 kenter user   813 Nov  6 16:23 .project
-rwxrwx--- 1 kenter user  3156 Nov  6 16:23 utility.cpp
-rwxrwx--- 1 kenter user   452 Nov  6 16:23 utility.h

```

Starting from this, you can import this project into an empty (or existing) workspace as follows.

- Start eclipse.
- Close the welcome screen, if it shows up.
- Click *File -> Open Projects from File System*
- Select the Directory `Task1Exercise2SolutionCleanEclipseProject/SimpleOpenCL`
- Click *Ok* and *Finish*
- You can switch to the perspective that you have worked with in Task1 clicking *Window -> Perspective -> Open Perspective -> C/C++*.
- From there on, your project behaves like the one you have used in Task1.

Feel free to use `Task1Exercise2SolutionCleanEclipseProject`, including it's project settings and the `.gitignore` as template for future Eclipse projects (you will need to modify them when working with the 18.x OpenCL SDK and Stratix 10 FPGAs).

### Troubleshooting

When your Eclipse environment stops working, you may be able to recover by deleting the workspace and settings, but keeping the project files. E.g., first close Eclipse, then type

```
rm -r .metadata/
rm -r SimpleOpenCL/.settings/
```

From this state, you should be able to import the project into a new workspace with the steps from the previous section.
