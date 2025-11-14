# load_to_neo4j_async_batch.py
import json
import asyncio
from neo4j import AsyncGraphDatabase
from tqdm.asyncio import tqdm
import config

DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 100  # Adjust batch size for your dataset

async def create_constraints(tx):
    await tx.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
    )

async def upsert_nodes_batch(tx, nodes):
    query = """
    UNWIND $nodes AS node
    MERGE (n:Entity {id: node.id})
    SET n += node.props
    """
    node_list = [
        {"id": node["id"], "props": {k: v for k, v in node.items() if k != "connections"}}
        for node in nodes
    ]
    await tx.run(query, nodes=node_list)

async def create_relationships_batch(tx, relationships):
    query = """
    UNWIND $rels AS r
    MATCH (a:Entity {id: r.source_id}), (b:Entity {id: r.target_id})
    MERGE (a)-[rel:RELATED_TO]->(b)
    SET rel += r.props
    """
    await tx.run(query, rels=relationships)

async def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    # Prepare relationships
    relationships = []
    for node in nodes:
        for rel in node.get("connections", []):
            if "target" in rel:
                relationships.append({
                    "source_id": node["id"],
                    "target_id": rel["target"],
                    "props": {k: v for k, v in rel.items() if k not in ["target", "relation"]}
                })

    driver = AsyncGraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )

    async with driver:
        async with driver.session() as session:
            await session.execute_write(create_constraints)

            # Batch insert nodes
            for i in tqdm(range(0, len(nodes), BATCH_SIZE), desc="Creating nodes"):
                batch = nodes[i:i + BATCH_SIZE]
                await session.execute_write(upsert_nodes_batch, batch)

            # Batch insert relationships
            for i in tqdm(range(0, len(relationships), BATCH_SIZE), desc="Creating relationships"):
                batch = relationships[i:i + BATCH_SIZE]
                await session.execute_write(create_relationships_batch, batch)

    print("Done loading into Neo4j.")

if __name__ == "__main__":
    asyncio.run(main())
