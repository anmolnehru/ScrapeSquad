# Path: app.core.langchain.py

"""
Client for interacting with the Langchain API.
"""
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core import conf

llm = conf.openai.get_model()

str_output_parser = StrOutputParser()


async def extract_text_from_url(url: str) -> str:
    loader = AsyncChromiumLoader([url])
    transformer = BeautifulSoupTransformer()
    documents = await loader.aload()
    documents_transformed = transformer.transform_documents(documents)
    return documents_transformed[0].page_content


def generate_cover_letter(profile, job, template) -> str:
    generation_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant that helps users generate cover letters for job applications.",
            ),
            ("user", "Hello, tailor my resume based on my background: {profile}"),
            (
                "system",
                """Great, please provide me with a job desctiption and I will generate a cover letter from this template: {template}.""",
            ),
            ("user", "Awesome! Here are the job details:\n{job}"),
        ]
    )
    chain = generation_template | llm | str_output_parser
    return chain.invoke({"profile": profile, "job": job, "template": template})


def generate_resume(profile, job, template) -> str:
    generation_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant that helps users generate resumes for job applications.",
            ),
            ("user", "Hello, tailor my resume based on my background: {profile}"),
            (
                "system",
                """Great, please provide me with a job desctiption and I will generate a resume from this template: {template}.""",
            ),
            ("user", "Awesome! Here are the job details:\n{job}"),
        ]
    )
    chain = generation_template | llm | str_output_parser
    return chain.invoke({"profile": profile, "job": job, "template": template})
