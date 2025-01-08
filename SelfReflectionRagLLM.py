from typing import List, Optional, Any
from pydantic import BaseModel, Field
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import hub
import torch
import json


class LLMModel:
    """Singleton class to manage the LLM model"""
    _instance = None

    def __new__(cls, model_dir: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_model(model_dir)
        return cls._instance

    def initialize_model(self, model_dir: str):
        """Initialize model, tokenizer, and pipeline"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=2048,
            temperature=0.01,
            top_p=0.95,
            repetition_penalty=1.15,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)


class BaseGrade(BaseModel):
    """Base class for all grading models"""
    binary_score: str = Field(description="Base binary score field")


class GradeDocuments(BaseGrade):
    """Grading model for document relevance"""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucinations(BaseGrade):
    """Grading model for hallucinations"""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class GradeAnswer(BaseGrade):
    """Grading model for answer assessment"""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


class RAGSystem:
    def __init__(self, model_dir: str):
        """Initialize RAG system with model and retriever"""
        self.llm_model = LLMModel(model_dir)
        self.retriever = None
        self.initialize_parsers()
        self.initialize_prompts()

    def set_retriever(self, retriever):
        """Set the retriever instance"""
        self.retriever = retriever

    def get_relevant_documents(self, question: str):
        """Compatibility method for older code"""
        return self.retriever.invoke(question)

    def invoke(self, *args, **kwargs):
        """Compatibility method for invoke-style calls"""
        if 'context' in kwargs and 'question' in kwargs:
            return self.generate_answer(kwargs['question'], kwargs['context'])
        elif 'question' in kwargs and 'document' in kwargs:
            return self.grade_document(kwargs['question'], kwargs['document'])
        elif 'documents' in kwargs and 'generation' in kwargs:
            return self.check_hallucination(kwargs['documents'], kwargs['generation'])
        elif 'question' in kwargs:
            return self.rewrite_question(kwargs['question'])
        else:
            raise ValueError("Invalid arguments for invoke method")

    def initialize_parsers(self):
        """Initialize all parsers"""
        self.doc_parser = PydanticOutputParser(pydantic_object=GradeDocuments)
        self.hallucination_parser = PydanticOutputParser(pydantic_object=GradeHallucinations)
        self.answer_parser = PydanticOutputParser(pydantic_object=GradeAnswer)

    def initialize_prompts(self):
        """Initialize all prompts"""
        # Retrieval grader prompt
        self.retrieval_prompt = self._create_grading_prompt(
            "You are a grader evaluating document relevance.",
            "Document: {document}\nQuestion: {question}",
            self.doc_parser
        )

        # Hallucination grader prompt
        self.hallucination_prompt = self._create_grading_prompt(
            "You are a grader assessing whether an LLM generation is grounded in retrieved facts.",
            "Set of facts:\n{documents}\n\nLLM generation:\n{generation}",  # Simplified template
            self.hallucination_parser
        )

        # Answer grader prompt
        self.answer_prompt = self._create_grading_prompt(
            "You are a grader assessing whether an answer addresses the question.",
            "User question:\n{question}\n\nLLM generation:\n{generation}",  # Simplified template
            self.answer_parser
        )

        # Question rewriter prompt
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a question re-writer optimizing for vectorstore retrieval. Return only the rewritten question."),
            ("human", "Here is the initial question:\n{question}\nFormulate an improved question.")
        ])

    def _create_grading_prompt(self, system_intro: str, human_template: str,
                               parser: PydanticOutputParser) -> ChatPromptTemplate:
        """Helper to create grading prompts"""
        system_template = f"""{system_intro}
        Return only a JSON response in this format: {{"binary_score": "yes"}} or {{"binary_score": "no"}}"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

    def _process_json_response(self, llm_output: str, grade_class: Any) -> Optional[BaseGrade]:
        """Process JSON response from LLM"""
        try:
            # First try to find any JSON-like structure in the output
            text = llm_output.strip()
            start_idx = text.find("{")
            end_idx = text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                try:
                    json_str = text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return grade_class(binary_score=result.get("binary_score", "no"))
                except:
                    pass

            # If no valid JSON found, fall back to text analysis
            text_lower = text.lower()
            if "yes" in text_lower:
                return grade_class(binary_score="yes")
            elif "no" in text_lower:
                return grade_class(binary_score="no")

            # Default fallback
            return grade_class(binary_score="no")

        except Exception as e:
            print(f"Error processing response: {e}")
            return grade_class(binary_score="no")  # Safe default

    def grade_document(self, question: str, document: str) -> Optional[GradeDocuments]:
        """Grade document relevance"""
        try:
            result = self.llm_model.llm.invoke(
                self.retrieval_prompt.format(
                    question=question,
                    document=document
                )
            )
            return self._process_json_response(result, GradeDocuments)
        except Exception as e:
            print(f"Error in grade_document: {e}")
            return None

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using hybrid approach (fine-tuned model + RAG)"""
        # Get direct answer from fine-tuned model
        base_answer = self.llm_model.llm.invoke(question)

        # If we have context, enhance the answer
        if context:
            # Create enhanced prompt
            rag_prompt = hub.pull("rlm/rag-prompt")
            enhanced_prompt = {
                "context": f"Base Knowledge: {base_answer}\nAdditional Context: {context}",
                "question": question
            }
            # Get enhanced answer
            chain = rag_prompt | self.llm_model.llm | StrOutputParser()
            return chain.invoke(enhanced_prompt)
        """Generate answer using RAG"""
        # rag_prompt = hub.pull("rlm/rag-prompt")
        # chain = rag_prompt | self.llm_model.llm | StrOutputParser()
        # return chain.invoke({"context": context, "question": question})
        # If no context, return base answer
        return base_answer

    def check_hallucination(self, documents: List[str], generation: str) -> Optional[GradeHallucinations]:
        """Check for hallucinations"""
        try:
            # Format the documents into a single string
            docs_text = "\n\n".join(documents)

            # Get LLM response
            result = self.llm_model.llm.invoke(
                self.hallucination_prompt.format(
                    documents=docs_text,
                    generation=generation
                )
            )

            # Process the response
            grading_result = self._process_json_response(result, GradeHallucinations)
            print(f"Hallucination check result: {grading_result.binary_score if grading_result else 'Error'}")
            return grading_result
        except Exception as e:
            print(f"Error in hallucination check: {e}")
            return None

    def grade_answer(self, question: str, generation: str) -> Optional[GradeAnswer]:
        """Grade answer quality"""
        try:
            prompt = self.answer_prompt.format(
                question=question,
                generation=generation
            )  # Removed format_instructions parameter
            result = self.llm_model.llm.invoke(prompt)
            return self._process_json_response(result, GradeAnswer)
        except Exception as e:
            print(f"Error in grade_answer: {e}")
            return None

    def rewrite_question(self, question: str) -> str:
        """Rewrite question for better retrieval"""
        chain = self.rewrite_prompt | self.llm_model.llm | StrOutputParser()
        try:
            return chain.invoke({"question": question})
        except Exception as e:
            print(f"Error rewriting question: {e}")
            return question


def main():
    # Initialize the system
    model_dir = "/home/ali/moradi/models/Radman-Llama-3.2-3B/extra"
    rag_system = RAGSystem(model_dir)

    # Initialize and set retriever - Updated integration
    from SelfReflectionRagRetriever import DocumentRetriever
    doc_retriever = DocumentRetriever(model_dir)
    rag_system.set_retriever(doc_retriever.get_retriever())

    # Example workflow
    original_question = ("How can i register in the conference?")

    # 1. Rewrite question
    question = rag_system.rewrite_question(original_question)
    print(f"Original question: {original_question}")
    print(f"Rewritten question: {question}")

    # 2. Get documents using invoke
    docs = rag_system.retriever.invoke(question)

    # 3. Grade documents
    doc_contents = [doc.page_content for doc in docs]
    for doc in doc_contents:
        relevance = rag_system.grade_document(question, doc)
        print(f"Document relevance: {relevance.binary_score if relevance else 'Error'}")

    # 4. Generate answer
    context = "\n\n".join(doc_contents)
    generation = rag_system.generate_answer(question, context)
    print(f"Generated answer: {generation}")

    # 5. Check for hallucinations with better error handling
    hallucination_result = rag_system.check_hallucination(doc_contents, generation)
    if hallucination_result:
        print(f"Hallucination check: {hallucination_result.binary_score}")

    # 6. Grade answer
    answer_grade = rag_system.grade_answer(question, generation)
    if answer_grade:
        print(f"Answer grade: {answer_grade.binary_score}")


if __name__ == "__main__":
    main()