import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_groq import ChatGroq
# load_dotenv()

# print("URI:", os.getenv("NEO4J_URI"))
# print("USER:", os.getenv("NEO4J_USERNAME"))
# print("PASS:", os.getenv("NEO4J_PASSWORD"))

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME= os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


st.set_page_config(page_title="🎬 Movie Graph Chatbot")
st.title("🎬 Movie Graph Chatbot")
# st.caption = ("NEO4J + Langchain+Streamlit")

@st.cache_data
def load_csv():
    return pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv"
    )

df = load_csv()

driver = GraphDatabase.driver(
    NEO4J_URI,auth=(NEO4J_USERNAME,NEO4J_PASSWORD)
)

def create_graph(tx,row):
    tx.run("""
    MERGE (m:Movie {id: $movieId})
    SET m.title = $title,
        m.released = $released,
        m.rating = $rating

    FOREACH (actor IN $actors |
        MERGE (a:Actor {name: actor})
        MERGE (a)-[:ACTED_IN]->(m)
    )

    MERGE (d:Director {name: $director})
    MERGE (d)-[:DIRECTED]->(m)

    FOREACH (genre IN $genres |
        MERGE (g:Genre {name: genre})
        MERGE (m)-[:HAS_GENRE]->(g)
    )
    """,
           
    movieId = row.movieId,
    title = row.title,
    released = int(row.released.split("-")[0]),
    rating = float(row.imdbRating),
    actors=[a.strip() for a in str(row.actors).split("|")],
    director = row.director,
    genres=[g.strip() for g in str(row.genres).split("|")]  
    )
if st.button("Load Data into NEO4J"):
    with driver.session() as session:
        for _,row in df.iterrows():
            session.execute_write(create_graph,row)
    st.success("✅ Graph data loaded successfully!")


## langchain GRaph QA setup

graph = Neo4jGraph(
    url = NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

question = st.text_input("Ask a movie-related question")

if question:
    with st.spinner("Thinking..."):
        answer = chain.run(question)
        st.success(answer)