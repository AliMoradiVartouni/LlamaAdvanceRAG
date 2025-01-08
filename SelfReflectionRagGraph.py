import os
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
from pprint import pprint
from SelfReflectionRagLLM import RAGSystem
from SelfReflectionRagRetriever import DocumentRetriever


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    attempts: int  # Add this


def create_graph(rag_system: RAGSystem):
    ################################################# Nodes  ################################
    def retrieve(state):
        """Retrieve documents"""
        print("---RETRIEVE---")
        question = state["question"]
        documents = rag_system.retriever.invoke(question)

        print(f"\nQuestion: {question}")
        print(f"Number of documents retrieved: {len(documents)}")
        print("\nDocument Contents:")
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print("-" * 50)
            print(doc.page_content)
            print("-" * 50)

        return {"documents": documents, "question": question, "attempts": state.get("attempts", 0)}

    def generate(state):
        """Generate answer"""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            print("No documents available - Generating general response")
            return {
                "documents": documents,
                "question": question,
                "generation": "I don't have enough specific information to answer that question. Could you please rephrase or ask something else?"
            }

        context = "\n\n".join(doc.page_content for doc in documents)
        generation = rag_system.generate_answer(question, context)
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """Grade document relevance"""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for doc in documents:
            relevance = rag_system.grade_document(question, doc.page_content)
            if relevance and relevance.binary_score == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        # If no relevant docs found, keep at least the top one
        if not filtered_docs and documents:
            print("---NO RELEVANT DOCS FOUND, KEEPING TOP DOCUMENT---")
            filtered_docs.append(documents[0])

        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """Transform query"""
        print("---TRANSFORM QUERY---")
        question = state["question"]
        better_question = rag_system.rewrite_question(question)
        return {"documents": state["documents"], "question": better_question}

    ######################################### Edges ######################################
    def decide_to_generate(state):
        """Decide next step based on document relevance"""
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        attempts = state.get("attempts", 0)

        print(f"Attempts so far: {attempts}")
        print(f"Number of filtered documents: {len(filtered_documents)}")

        # If we've tried too many times, force generation
        if attempts >= 2:
            print("MAX ATTEMPTS REACHED - Forcing generation with all available documents")
            # Use all documents instead of filtered ones
            state["documents"] = state.get("original_documents", filtered_documents)
            return "generate"

        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, TRANSFORM QUERY---")
            state["attempts"] = attempts + 1
            return "transform_query"

        print("---DECISION: GENERATE---")
        return "generate"

    def grade_generation(state, rag_system):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state
            rag_system: The RAG system instance

        Returns:
            str: Decision for next node to call
        """
        print("---CHECK GENERATION---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        try:
            print("---CHECK HALLUCINATIONS---")
            hallucination_check = rag_system.check_hallucination(documents, generation)

            if hallucination_check and hallucination_check.binary_score == "yes":
                print("---GENERATION GROUNDED IN DOCUMENTS - CHECKING ANSWER---")
                answer_check = rag_system.grade_answer(question, generation)

                if answer_check and answer_check.binary_score == "yes":
                    print("---GENERATION ADDRESSES QUESTION - ACCEPTING---")
                    return "useful"

                print("---GENERATION DOES NOT ADDRESS QUESTION - TRYING AGAIN---")
                return "not useful"

            print("---GENERATION NOT GROUNDED - RETRYING---")
            return "not supported"

        except Exception as e:
            print(f"---ERROR IN GRADING: {e}---")
            print("---FORCING ACCEPTANCE DUE TO ERROR---")
            return "useful"

    ################################ Build Graph ################################
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "generate",
        lambda x: grade_generation(x, rag_system),
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()


def main():
    try:
        model_dir = "/home/ali/moradi/models/Radman-Llama-3.2-3B/extra"
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")

        # Initialize RAG system
        rag_system = RAGSystem(model_dir)
        retriever = DocumentRetriever(model_dir)
        rag_system.set_retriever(retriever.get_retriever())

        # Create and compile graph
        app = create_graph(rag_system)

        print("\nBot is ready! Type 'quit' to exit.")
        print("-" * 50)

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                break

            # Initialize state with user question
            state = {
                "question": user_input,
                "attempts": 0
            }

            # Process through graph
            for output in app.stream(state):
                for key, value in output.items():
                    pprint(f"Node '{key}':")
                pprint("\n---\n")

            # Show final result
            pprint(f"Final answer: {value.get('generation', 'No answer generated')}")
            print("-" * 50)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check configuration and try again.")


if __name__ == "__main__":
    main()