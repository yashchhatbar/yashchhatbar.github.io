export const projects = [

        {
                slug: "face-deduplication",
                title: "Face De-duplication & Authentication",
                image: "/images/project/face-1.png",
                hero: "/images/project/face-1.png",
                year: "2025",

                tech: ["Python", "OpenCV", "FaceNet", "Deep Learning", "CNN", "NumPy", "Scikit-learn"],

                problem: `
Many digital platforms such as financial services, government portals,
and identity verification systems suffer from duplicate identities.

A single individual can create multiple accounts using slightly different
personal information, making it difficult for traditional systems to detect duplicates.

Duplicate records lead to several issues including fraud, inaccurate analytics,
security risks, and poor database integrity.

Most identity verification systems rely on manual verification or document-based
validation which is slow, expensive, and prone to human error.

The challenge was to design an automated AI-based system that can determine
whether a newly registered user already exists in the database by analyzing
their facial features.
`,

                solution: `
To solve this problem, I built an AI-powered facial recognition system that
detects duplicate identities using deep learning.

The system captures a user's face image and extracts facial embeddings using
a convolutional neural network model.

Facial embeddings are numerical vectors representing unique facial characteristics.

These embeddings are compared against existing user embeddings stored in the database.

Cosine similarity is used to calculate how closely two faces match.

If the similarity score exceeds a threshold value, the system flags the new
registration as a potential duplicate identity.

This automated approach enables fast, scalable, and reliable identity verification
without manual intervention.
`,

                impact: `
The AI-based system significantly improves identity verification processes.

Key benefits include:
• Automated duplicate identity detection  
• Faster onboarding for legitimate users  
• Improved database accuracy  
• Reduced risk of identity fraud  
• Scalable solution for large user bases  

This project demonstrates how biometric AI systems can enhance security
and trust in digital identity platforms.
`,

                github: "https://github.com/yashchhatbar",

                gallery: [
                        "/images/project/face-2.png",
                        "/images/project/face-3.png",
                        "/images/project/face-4.png"
                ]
        },

        {
                slug: "enterprise-ai-assistant",
                title: "Enterprise AI Assistant",
                image: "/images/project/assistant-1.png",
                hero: "/images/project/assistant-1.png",
                year: "2025",

                tech: ["Python", "LangChain", "Vector DB", "RAG", "OpenAI API"],

                problem: `
Modern organizations store large amounts of internal knowledge across
documents, manuals, research papers, and company databases.

Employees often struggle to quickly find the information they need,
which leads to wasted time and reduced productivity.

Traditional search systems rely on keyword matching which frequently
returns irrelevant or incomplete results.

This forces employees to manually review multiple documents before
finding the correct information.

As organizations grow, this inefficiency becomes a major productivity bottleneck.

The challenge was to design an AI assistant capable of understanding
natural language questions and retrieving accurate answers from internal
enterprise knowledge sources.
`,

                solution: `
To address this problem, I developed an Enterprise AI Assistant
using Retrieval Augmented Generation (RAG).

Company documents are first converted into semantic embeddings
using language models and stored in a vector database.

When a user asks a question, semantic search retrieves the most
relevant documents based on meaning rather than keywords.

These documents are then provided as context to a large language model
which generates an accurate and contextual response.

The system integrates Python, LangChain, and vector databases to
manage document indexing, retrieval, and response generation.

This architecture enables employees to interact with enterprise
knowledge using natural language conversations.
`,

                impact: `
The AI assistant dramatically improves knowledge accessibility
within organizations.

Key outcomes include:
• Faster access to company knowledge  
• Reduced time spent searching documents  
• Improved employee productivity  
• Enhanced collaboration across teams  
• Intelligent enterprise knowledge management  

This system demonstrates how AI assistants can transform
enterprise workflows and improve operational efficiency.
`,

                github: "https://github.com/yashchhatbar",

                gallery: [
                        "/images/project/assistant-2.png",
                        "/images/project/assistant-3.png",
                        "/images/project/assistant-4.png"
                ]
        },

        {
                slug: "multi-agent-ai",
                title: "Multi-Agent AI Research System",
                image: "/images/project/agent-1.png",
                hero: "/images/project/agent-1.png",
                year: "2025",

                tech: ["Python", "LLM", "LangGraph", "AI Agents"],

                problem: `
Research workflows often require gathering information from multiple
sources, analyzing data, summarizing findings, and generating insights.

Performing these tasks manually is time-consuming and repetitive.

Researchers frequently repeat the same processes such as searching
for information, extracting key points, and synthesizing knowledge.

When dealing with large datasets or complex research topics,
these workflows become inefficient and difficult to manage.

The challenge was to design an AI architecture capable of automating
research workflows by coordinating multiple intelligent agents
that collaborate to complete tasks autonomously.
`,

                solution: `
The solution was to build a multi-agent AI research system where
different agents specialize in specific tasks.

One agent performs information retrieval from documents or web sources.

Another agent analyzes collected data and extracts key insights.

A third agent summarizes the information into structured outputs
such as reports or research summaries.

These agents communicate with each other through a shared memory
and orchestration layer.

Large language models provide reasoning capabilities while Python
manages workflow coordination.

This collaborative architecture enables agents to divide complex
problems into smaller tasks and solve them efficiently.
`,

                impact: `
The multi-agent research system demonstrates the potential of
autonomous AI agents in complex knowledge workflows.

Key benefits include:
• Automated research and data collection  
• Faster knowledge synthesis  
• Reduced manual workload for researchers  
• Scalable architecture for large research tasks  

This project highlights the future potential of collaborative AI
agents for automating knowledge-intensive processes.
`,

                github: "https://github.com/yashchhatbar/multi-agent-research",

                gallery: [
                        "/images/project/agent-2.png",
                        "/images/project/agent-3.png",
                        "/images/project/agent-4.png"
                ]
        },

        {
                slug: "docgenius",
                title: "DocGenius – AI PDF Assistant",
                image: "/images/project/pdf-1.png",
                hero: "/images/project/pdf-1.png",
                year: "2025",

                tech: ["Python", "LLM", "NLP", "Vector Search"],

                problem: `
Professionals frequently work with large documents such as research papers,
technical manuals, and legal reports that contain hundreds of pages.

Extracting useful insights from these documents requires manually reading
large volumes of text which is extremely time consuming.

Traditional document search tools rely on keyword matching which often fails
to understand the real context of the content.

Users may need to scan multiple sections of a document before locating the
information they actually need.

The challenge was to build an AI system that could understand the semantic
content of documents and allow users to interact with them using natural
language questions.
`,

                solution: `
To solve this problem, I developed DocGenius, an AI-powered document assistant.

The system first extracts text from uploaded PDF files and splits the content
into smaller semantic chunks.

These document chunks are converted into embeddings using language models
and stored in a vector database.

When a user asks a question, the system performs semantic search to retrieve
the most relevant document sections.

Those sections are then passed to a large language model which generates
a context-aware answer based on the document content.

This architecture combines NLP pipelines, vector search, and language models
to create an interactive AI-powered document exploration system.
`,

                impact: `
DocGenius transforms static PDF documents into interactive knowledge sources.

Key benefits include:
• Instant answers from large documents  
• Reduced manual reading time  
• Improved research productivity  
• Better document understanding  

The system demonstrates how AI-powered document assistants can help
professionals interact with information more efficiently.
`,

                github: "https://github.com/yashchhatbar/DocGenius-Revolutionizing-PDFs-with-AI",

                gallery: [
                        "/images/project/pdf-2.png",
                        "/images/project/pdf-3.png",
                        "/images/project/pdf-4.png"
                ]
        },

        {
                slug: "voice-ordering",
                title: "AI Voice Ordering System",
                image: "/images/project/voice-1.png",
                hero: "/images/project/voice-1.png",
                year: "2025",

                tech: ["Python", "Speech Recognition", "NLP"],

                problem: `
Restaurants and retail stores often experience operational inefficiencies
during peak hours when large numbers of customers place orders.

Manual order taking requires staff members to listen, record orders,
and input them into the system which slows down service.

Communication errors between customers and staff may also lead to
incorrect orders and reduced customer satisfaction.

As businesses grow, this manual process becomes difficult to scale.

The challenge was to design an AI-powered system that allows customers
to place orders using voice commands without requiring manual input
from staff members.
`,

                solution: `
The AI Voice Ordering System enables customers to interact with an
ordering system using natural speech.

Speech recognition technology converts spoken input into text
which is then processed using natural language processing techniques.

The NLP pipeline extracts order information such as items, quantities,
and special instructions.

The interpreted order is automatically formatted and sent to the
restaurant’s ordering system.

The architecture integrates speech recognition models with Python-based
NLP processing to interpret customer intent accurately.

This approach allows businesses to automate order intake and reduce
human dependency in the ordering workflow.
`,

                impact: `
The system significantly improves customer service efficiency.

Key outcomes include:
• Faster ordering process  
• Reduced workload for staff  
• Improved order accuracy  
• Shorter waiting times for customers  

This project highlights the potential of conversational AI
in automating service industry workflows.
`,

                github: "https://github.com/yashchhatbar/Embeddable-AI-Voice-Ordering-Template",

                gallery: [
                        "/images/project/voice-2.png",
                        "/images/project/voice-3.png",
                        "/images/project/voice-4.png"
                ]
        },

        {
                slug: "chest-disease",
                title: "Chest Disease Classification",
                image: "/images/project/chest-1.png",
                hero: "/images/project/chest-1.png",
                year: "2025",

                tech: ["TensorFlow", "CNN", "Deep Learning"],

                problem: `
Diagnosing chest diseases such as pneumonia and tuberculosis often
requires analysis of medical X-ray images by trained radiologists.

In many healthcare systems, the number of medical images exceeds
the available diagnostic resources.

Manual analysis of X-ray images can be slow and is sometimes affected
by human error.

Delayed detection of diseases may negatively impact treatment outcomes.

The challenge was to develop an AI model capable of automatically
analyzing chest X-ray images and identifying disease patterns
with high accuracy.
`,

                solution: `
To address this challenge, I built a deep learning model using
convolutional neural networks.

The system was trained on labeled chest X-ray datasets containing
examples of different chest diseases.

Image preprocessing techniques such as normalization, resizing,
and augmentation were applied to improve training quality.

The CNN architecture learns visual patterns associated with
specific disease indicators during training.

Once trained, the model can analyze new X-ray images and classify
the detected condition automatically.

TensorFlow was used to implement the training pipeline and
optimize model performance.
`,

                impact: `
The AI model can support healthcare professionals by assisting
with medical image analysis.

Key benefits include:
• Faster disease detection  
• Reduced diagnostic workload for radiologists  
• Early identification of abnormalities  
• Improved healthcare efficiency  

This project demonstrates how AI can enhance medical diagnostics
and support clinical decision-making.
`,

                github: "https://github.com/yashchhatbar/Chest-Disease-Classification",

                gallery: [
                        "/images/project/chest-2.png",
                        "/images/project/chest-3.png",
                        "/images/project/chest-4.png"
                ]
        },

        {
                slug: "digit-recognizer",
                title: "Handwritten Digit Recognizer",
                image: "/images/project/digits-1.png",
                hero: "/images/project/digits-1.png",
                year: "2025",

                tech: ["Python", "TensorFlow", "CNN", "MNIST"],

                problem: `
Recognizing handwritten digits is an important task in industries
such as banking, postal services, and document digitization.

Manual interpretation of handwritten numbers becomes inefficient
when processing large volumes of documents.

Handwriting styles vary significantly between individuals which
makes rule-based recognition systems unreliable.

The challenge was to develop a machine learning system capable
of accurately identifying handwritten digits from image data.
`,

                solution: `
The solution involved training a convolutional neural network
using the MNIST handwritten digit dataset.

The dataset contains thousands of labeled digit images which
allow the model to learn visual patterns associated with each number.

Images are first preprocessed through normalization and resizing
before being fed into the neural network.

The CNN architecture extracts hierarchical features from images
and uses them to classify digits.

TensorFlow was used to train and evaluate the model performance.

After training, the system can predict handwritten digits with
high accuracy.
`,

                impact: `
The project demonstrates how deep learning models can automate
image recognition tasks.

Key outcomes include:
• Accurate handwritten digit recognition  
• Reduced manual document processing  
• Improved automation in digit-based workflows  

The system highlights the effectiveness of CNN models in
computer vision applications.
`,

                github: "https://github.com/yashchhatbar/Handwritten-digit-recognizer",

                gallery: [
                        "/images/project/digits-2.png",
                        "/images/project/digits-3.png",
                        "/images/project/digits-4.png"
                ]
        },

        {
                slug: "resume-screening",
                title: "AI Resume Screening System",
                image: "/images/project/resume-1.png",
                hero: "/images/project/resume-1.png",
                year: "2025",

                tech: ["Python", "NLP", "Machine Learning"],

                problem: `
Recruitment teams often receive hundreds of resumes for a single
job position.

Manually reviewing each resume to identify qualified candidates
requires significant time and effort.

Human reviewers may overlook strong candidates or spend time
reviewing applications that do not meet job requirements.

Traditional keyword filtering systems also struggle to capture
the full context of candidate experience.

The challenge was to design an AI system capable of automatically
analyzing resumes and ranking candidates based on their relevance
to a specific job description.
`,

                solution: `
The AI Resume Screening System uses natural language processing
to analyze resume content.

Key information such as skills, education, and experience is
extracted from resume text.

These features are compared with job descriptions using similarity
analysis techniques.

Machine learning algorithms rank candidates based on how closely
their profiles match the required qualifications.

Python NLP libraries were used to preprocess resume data and
extract structured information.

The system automatically generates candidate rankings for recruiters.
`,

                impact: `
The system significantly improves recruitment efficiency.

Key benefits include:
• Automated resume analysis  
• Faster candidate shortlisting  
• Reduced manual screening workload  
• Data-driven hiring decisions  

This project demonstrates how AI can streamline recruitment
processes and improve talent acquisition workflows.
`,

                github: "https://github.com/yashchhatbar",

                gallery: [
                        "/images/project/resume-2.png",
                        "/images/project/resume-3.png",
                        "/images/project/resume-4.png",
                ]
        },

        {
                slug: "movie-recommendation",
                title: "Movie Recommendation System",
                image: "/images/project/movie-1.png",
                hero: "/images/project/movie-1.png",
                year: "2025",

                tech: ["Python", "Machine Learning", "Collaborative Filtering"],

                problem: `
Streaming platforms host thousands of movies and TV shows,
making it difficult for users to discover content that matches
their personal preferences.

Without personalized recommendations, users may spend excessive
time browsing through content libraries.

Traditional recommendation systems based on popularity rankings
do not fully capture individual user interests.

The challenge was to design a recommendation system that can
analyze user behavior and generate personalized movie suggestions.
`,

                solution: `
The movie recommendation system was developed using collaborative
filtering techniques.

User ratings and viewing histories were analyzed to identify
patterns in user preferences.

Similarity metrics were used to identify users with similar tastes.

Movies liked by similar users were recommended to others
with comparable preferences.

Python data analysis libraries were used to preprocess datasets
and implement the recommendation algorithms.

The system dynamically generates personalized movie suggestions
based on historical user interactions.
`,

                impact: `
The recommendation engine improves content discovery for users.

Key outcomes include:
• Personalized movie recommendations  
• Improved user engagement  
• Reduced browsing time  

This project demonstrates how machine learning can enhance
user experience on content platforms.
`,

                github: "https://github.com/yashchhatbar/movie-recommendation",

                gallery: [
                        "/images/project/movie-2.png",
                        "/images/project/movie-3.png",
                        "/images/project/movie-4.png",
                ]
        },

        {
                slug: "customer-segmentation",
                title: "Customer Segmentation",
                image: "/images/project/customer-1.png",
                hero: "/images/project/customer-1.png",
                year: "2025",

                tech: ["Python", "K-Means", "Data Analysis"],

                problem: `
Businesses collect large volumes of customer data but often
struggle to derive meaningful insights from it.

Without understanding customer behavior, companies cannot
design effective marketing strategies.

Treating all customers as a single group results in inefficient
marketing campaigns and lower conversion rates.

The challenge was to analyze customer purchasing behavior and
identify meaningful customer segments.
`,

                solution: `
Customer segmentation was performed using the K-Means clustering
algorithm.

Customer purchase data and behavioral attributes were analyzed
to identify patterns.

The dataset was cleaned and normalized before applying clustering
techniques.

K-Means grouped customers into clusters representing different
types of purchasing behaviors.

Python data analysis libraries such as Pandas and Scikit-learn
were used to build and evaluate the clustering model.
`,

                impact: `
Customer segmentation enables businesses to design targeted
marketing campaigns.

Key outcomes include:
• Improved customer insights  
• More effective marketing strategies  
• Increased campaign conversion rates  

This project demonstrates how machine learning can help
businesses understand customer behavior.
`,

                github: "https://github.com/yashchhatbar/Customer-Segmentation-with-K-Means-Project",

                gallery: [
                        "/images/project/customer-2.png",
                        "/images/project/customer-3.png",
                        "/images/project/customer-4.png",
                ]
        },

        {
                slug: "sales-data-analysis",
                title: "Sales Data Analysis Dashboard",
                image: "/images/project/sales-1.png",
                hero: "/images/project/sales-1.png",
                year: "2025",

                tech: ["SQL", "Power BI", "Data Analysis"],

                problem: `
Businesses generate large volumes of sales data across
multiple regions, products, and time periods.

However, raw datasets are difficult to interpret without
proper analysis tools.

Decision makers often struggle to identify revenue trends,
top performing products, and regional sales patterns.

The challenge was to transform raw sales data into
interactive visual insights that help businesses make
better strategic decisions.
`,

                solution: `
The project involved cleaning and analyzing over 100,000
sales records using SQL and Python.

Data preprocessing removed inconsistencies and prepared
the dataset for analysis.

The cleaned dataset was then visualized using Power BI
to create an interactive dashboard.

The dashboard displays revenue trends, regional sales
performance, and product category insights.

Business users can explore the data dynamically to
identify patterns and opportunities.
`,

                impact: `
The dashboard enables data-driven business decision making.

Key benefits include:
• Clear visualization of revenue trends  
• Identification of top performing products  
• Improved business insights  

This project demonstrates the value of business
intelligence tools for strategic planning.
`,

                github: "https://github.com/yashchhatbar/ETL-Sales-Analysis-Report-MySQL-PowerBI-main",

                gallery: [
                        "/images/project/sales-2.png",
                        "/images/project/sales-3.png",
                        "/images/project/sales-4.png",
                ]
        },

        {
                slug: "todo-app",
                title: "To-Do List Application (PyQt)",
                image: "/images/project/todo-1.png",
                hero: "/images/project/todo-1.png",
                year: "2025",

                tech: ["Python", "PyQt", "SQLite"],

                problem: `
Managing daily tasks without a structured productivity
system can reduce efficiency and increase stress.

Many individuals rely on simple notes or memory
to track tasks which often leads to missed deadlines.

Existing task management tools may be complex
or require internet connectivity.

The challenge was to design a lightweight desktop
application that allows users to manage tasks easily
in a structured environment.
`,

                solution: `
The To-Do application was built using Python
and the PyQt framework.

Users can create, update, and delete tasks
through a graphical user interface.

Tasks can be assigned priorities and deadlines.

All task data is stored locally in a SQLite
database ensuring persistence between sessions.

The interface was designed to be simple,
minimal, and user friendly.
`,

                impact: `
The application improves productivity by
providing an organized way to manage tasks.

Key outcomes include:
• Improved task organization  
• Better time management  
• Increased personal productivity  

This project demonstrates how simple
desktop applications can significantly
improve workflow management.
`,

                github: "https://github.com/yashchhatbar",

                gallery: [
                        "/images/project/todo-2.png",
                        "/images/project/todo-3.png",
                        "/images/project/todo-4.png",
                ]
        }

]