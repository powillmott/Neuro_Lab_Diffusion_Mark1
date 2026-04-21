Project Summary:
This project is a decision based learning ai tool. It takes in a DBL tree and samples traversals of it, and uses those ground truths to generate practice problems. Users can then interact with the ai and traverse through it with the ai tutor.
ERD diagram for Supabase
<img width="1488" height="612" alt="image" src="https://github.com/user-attachments/assets/8636964d-6b59-4dc2-89e0-81f59a87c3ac" />
View of most of the graph database
<img width="554" height="465" alt="image" src="https://github.com/user-attachments/assets/0b6a0a2b-5f5c-4899-8dfc-687f8168870e" />

Video Demo (Part 1 and 2 split for file size):
https://github.com/user-attachments/assets/f46c57df-134d-42a2-861f-41cced3f68b7

https://github.com/user-attachments/assets/e27e3e1c-352c-4e68-a43d-96979e626954

I learned a ton about creating a full-stack application, first time really doing that, with users and authentication and data storage from scratch. I specifically learned a lot from using supabase, that was great learning the ways it works with authentication, and how to work with it to keep data consistent. Neo4j was a really fun part of the project as well, and there was one specific hard challenge getting users in it to match supabase, but some really cool supabase functions that creates a user in neo4j when it is created in supabase worked really well. Also, langchain and automatically created database tables. Still learning here, but it was cool to learn to have to work with others design choices. I had some early disconnect between some of my tables and the uses of theirs, redundancy that I was able to trim down on as I learned.

AI:
This project is heavily integrated with AI. Most of the working ai features are routed through langchain code, and it is able to connect with openai API token or through local llama models. This repo also contains deep learning ai models and training strategies for tracing trains of thought, and though I was not able to connect the pieces in time, is an ai feature of this project that I worked on. 

How AI was used to build this project:
copilot and gemini helped me code, it's amazing how quickly developers can scale while managing an ai "team" of novice programmers. Keeping bloat out and the project structure pure and purpose separated was my biggest focus when working with ai, because when I generated code (especially for the front end) it would sometimes want to incorporate logic that didn't belong there and that I likely already wrote where it did belong. But most of the line by line code was written by ai, and it was very helpful. 

Project Interest:
Education is a field that will experience a lot of challenges from emerging ai technology, but I feel like ai could be used to fix a lot of longstanding problems in education as well. I am passionate about education and the science of learning, and want to do research in modeling human deeper understanding using ai tools, and this project is a step in that direction. The lab that I am working in for parts of this project (and who owns part of the code, which is why this is in the deep learning repository, the other code is not solely mine, and so I cannot attatch that repository), is hoping that this project is a stepping stone to ai oral exams here at BYU. I hope to continue in this research for the foreseeable future.

This project can scale to the extent that it needs to easily, which is test groups of students in classes. BYU may choose to implement this research, but if they do they will take the science and ai features and implement them into their own architecture and authentication. For our testing purposes, supabase scales very nicely, users are handled correctly in both databases, concurrency isn't a problem for supabase, reading neo4j (which is all that users have the ability to do from the site), and instances of streamlit. Data will be collected into the database, and be used for research, and then taken down and moved onto new iterations.
