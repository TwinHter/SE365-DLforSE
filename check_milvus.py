"""Script để check Milvus collection."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pymilvus import MilvusClient

client = MilvusClient(uri='http://localhost:19530')

collection_name = 'rag_chunks'

if client.has_collection(collection_name):
    stats = client.get_collection_stats(collection_name)
    print(f'[OK] Collection "{collection_name}" ton tai')
    print(f'  So chunks: {stats.get("row_count", 0)}')

    results = client.query(
        collection_name=collection_name,
        output_fields=['chunk_id', 'category', 'year'],
        limit=3
    )
    print('\n  Sample chunks:')
    for r in results:
        print(f'    - {r}')
else:
    print(f'[X] Collection "{collection_name}" chua ton tai')
