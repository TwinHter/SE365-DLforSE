"""Script để xoa va embedding lai chunks vao Milvus."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import logging

from pymilvus import MilvusClient
from src.config import MILVUS_URI, MILVUS_COLLECTION
from src.rag_utils import ChunkDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    client = MilvusClient(uri=MILVUS_URI)
    
    # Xoa collection cu
    if client.has_collection(MILVUS_COLLECTION):
        print('Xoa collection cu...')
        client.drop_collection(MILVUS_COLLECTION)
        print(f'Da xoa collection "{MILVUS_COLLECTION}"')
    else:
        print('Collection chua ton tai')
    
    print('Dang load va embedding chunks...')
    db = ChunkDatabase()
    count = db.load()
    print(f'Da embedding va load {count} chunks vao Milvus')
    
    # Verify
    if client.has_collection(MILVUS_COLLECTION):
        stats = client.get_collection_stats(MILVUS_COLLECTION)
        print(f'Tong so chunks trong Milvus: {stats.get("row_count", "?")}')

if __name__ == '__main__':
    main()
