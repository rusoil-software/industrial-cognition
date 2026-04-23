# Project planning

## Where to find all project data

Project contains the plan in the following files:

1. [ideam.md](idea.md) contains formalized description of the project.
2. [vision.md](vision.md) contains full vision of the project and its parts
3. [conventions.md](./conventions.md) development conventions for every participant to follow
4. [workflow.md](./workflow.md) is a pipleine of development
5. [tasklist.md](./tasklist.md) is a formatted reference sheet of the project completion
6. [test-template.md](./test-templates.md) is a simple template for making tests
7. [Docker File](../Dockerfile)

## How to use those files

Follow the following rules when use the files from the links above:

- Always follow conventions described in [conventions.md](./conventions.md)
- Always first look in [vision.md](vision.md) to compare project progress to the vision
- Always adhere to the [workflow.md](./workflow.md) when doing anything.
- Always make tests after looking through the [test-template.md](./test-templates.md)
- Always update your progress in [tasklist.md](./tasklist.md)
- Always use the [Docker File](../Dockerfile) as main docker file, do not create others, make all changes there.
- Always start withg adding a file with description of a new technology or package to the [docs/](../doc/) folder whenever you add it to the project and provide full description of what exactly from this technology or package will be used and how with all proper references and explanations. Always name such file in style `<package-name>.md`.
- At the end of development (reaching the last stage on [tasklist.md](./tasklist.md)) always create file [onboarding.md](./onboarding.md) with links to all the technology files from above and instructions on how to start working with the project

## Project creation

Should the files listred above do not exist, you *must* create those by executing the prompts below in consequentila order:

**Prompt 1**:

```prompt
Please document my idea for developing an AI service described in the [Project Intro](./project-intro.md) in
the doc/idea.md file. AI (package will be named `industrial-cognition`) should be implemented in a form of a compact
AI service with a web control panel (FastAPI) for deployment on an on-site industrial server equipped with professional
industrial camera to detect positions of metal sheets, produced details and clutter in the robot's working zone.
The main purpose of the application (The machine vision module) should be a hardware and software system that takes
photographs using an industrial camera placed above the working zone of the robot (robotic arm), determines 
the coordinates of the objects on the images using OpenCV algorithms connected directly, and allows access to
this information online to other robotic systems. The AI should be able to run on a high-performance industrial computer. 
The AI processes the video stream from an industrial camera mounted above the robot's working zone, identifies objects
within the working zone based on predefined parameters like size threshold values etc., then sends response with 
asynchronous FastAPI route back to client via websocket connection so no additional requests are needed while 
processing, saving results into database.
```

**Prompt 2**:

```prompt
Let's create a @doc/vision.md file.
In it, we will reflect the technical vision of the @doc/idea.md project:
- technologies
- development
- project structure
- project architecture
- data model
- user interactions
- storage
- hardware requirements
- interaction with integratred toolset (cameras and laser)
- work scenarios
- deploy
- scaling
- approach to configuration
- approach to logging

This document will serve as our starting point and technical blueprint for further development.

Let's create sequentially.
Analyze the composition of the document.
Go through the sections sequentially.
Ask me questions, suggest your final vision, and after agreeing with me, record it in a document.
Then move on to the next section.

The most important:
We need to create the simplest possible solution to test our idea, according to the KISS principles.
No overengineering .
Only the most necessary and simple things in all sections!
```

**Prompt 3**:

```prompt
Create a docs/conventions.md file with rules for code development to pass to the code assistant, which will generate the code.
The rules should reflect all of our main development principles from the @doc/vision.md document and refer to the @doc/vision.md document itself, without duplicating information from it.
The rules should be concise, according to the KISS principle, not contain unnecessary things, only the main ones that affect the quality
```

**Prompt 4**:

```prompt
Create a step-by-step iterative development plan: doc/tasklist.md

Each step should allow you to test the bot's operation.
Each iteration adds new functionality.

At the top, set aside space for a progress report that will be updated after each iteration.
The report is beautiful in a table, with statuses and icons.

Each task should be marked with a checkbox to clearly track progress.
The plan should also be concise, contain only the main points and follow the KISS principle.
```

**Prompt 5**:

```prompt
Create a doc/workflow.md file with the rules for completing work on the tasklist @tasklist.md,
to instruct a code assistant on developing our bot on @vision.md

Important:
- carry out work strictly according to plan
- Before each iteration, first coordinate the proposed solution with code sections
- implement after approval
- after which wait for confirmation
- update progress in the tasklist
- mark completed tasks
- coordinate the transition to the next iteration
- make a commit to the repository

Workflow should be concise, contain only the essentials and follow the KISS principle.
```

**Prompt 6**:

```prompt
We're starting work on the @vision.md project strictly according to the @tasklist.md tasklist.
```

**Prompt 7**:

The API obtainment prompt (configre it to your specific needs before use):

```prompt
Create step-by-step instructions on how to get a SOME_API
Save it as a file in the / doc / guides folder
```

**Prompt 8**:

```prompt
Create a technical description of the developed project to quickly familiarize a new developer with the project.
Use not only text, but also code examples, links to files, diagrams, and other visualizations.
Save the description to the file doc/intro.md
```
