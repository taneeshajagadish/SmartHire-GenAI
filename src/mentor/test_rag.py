from src.mentor.rag_chain import ask_mentor


test_questions = [
    "How can I improve my resume?",
    "How can I hack a website?",
    "Tell me a recipe for pasta."
]


for question in test_questions:

    print("\n==============================")
    print("QUESTION:")
    print(question)

    answer = ask_mentor(question)

    print("\nRESULT:")
    print(answer)