from sqlalchemy import Column, String, Integer, DateTime, create_engine, select, make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime, timezone
from llama_index.core import SimpleDirectoryReader
import psycopg2

Base = declarative_base()

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    doc_id = Column(String, unique=True)  # LlamaIndex internal ID
    mime_type = Column(String)
    file_path = Column(String)
    size = Column(Integer)
    creation_date = Column(DateTime, default=datetime.now(timezone.utc))
    modified_at = Column(DateTime, default=datetime.now(timezone.utc))
    accessed_at = Column(DateTime, default=datetime.now(timezone.utc))
    # represents the filename and the llamaindex internal doc id
    def __repr__(self):
        return f"<DocumentMetadata(filename='{self.filename}', doc_id='{self.doc_id}')>"

# Database connection setup
# Reusing the same DB URL logic or just importing if possible, but for now standalone setup
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/vectordoc")

url = make_url(DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def search_document_by_doc_id(doc_id: str):
    """Search for a document metadata entry by its doc_id."""
    db = SessionLocal()
    try:
        document = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
        return document
    finally:
        db.close()

def search_document_by_filename(filename: str):
    """Search for a document metadata entry by its filename."""
    db = SessionLocal()
    try:
        document = db.query(DocumentMetadata).filter(DocumentMetadata.filename == filename).first()
        return document
    finally:
        db.close()

def check_document_exists_with_doc_id(doc_id: str) -> bool:
    """Check if a document with the given doc_id exists in the database."""
    db = SessionLocal()
    try:
        exists = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first() is not None
        return exists
    finally:
        db.close()

def check_document_exists_with_filename(filename: str) -> bool:
    """Check if a document with the given filename exists in the database."""
    db = SessionLocal()
    try:
        exists = db.query(DocumentMetadata).filter(DocumentMetadata.filename == filename).first() is not None
        return exists
    finally:
        db.close()

def get_doc_id_from_filename(filename: str) -> str | None:
    """Get the doc_id for a given filename."""
    db = SessionLocal()
    try:
        document = db.query(DocumentMetadata).filter(DocumentMetadata.filename == filename).first()
        if document:
            return document.doc_id
        return None
    finally:
        db.close()

def create_document_session(docs: SimpleDirectoryReader):
        """Creates a new SQLAlchemy session for document metadata operations."""
        db = SessionLocal()
        try:
            for doc in docs:
                # check if file is stored
                existing = check_document_exists_with_doc_id(doc.doc_id)
                # if we don't have the file, add it
                if not existing:
                    metadata = DocumentMetadata(
                        filename=doc.metadata.get("file_name", "unknown"),
                        doc_id=doc.doc_id,
                        mime_type=doc.metadata.get("file_type", "unknown"),
                        file_path=doc.metadata.get("file_path", "unknown"),
                        size=doc.metadata.get("file_size", 0),
                        creation_date=doc.metadata.get("creation_date", None),
                        modified_at=doc.metadata.get("modified_at", None),
                        accessed_at=doc.metadata.get("accessed_at", None),
                    )
                    db.add(metadata)
                # update document session
                else:
                    # use document id to update metadata in the table
                    stmt = select(DocumentMetadata).where(DocumentMetadata.doc_id == doc.doc_id)
                    document_to_update = db.scalar(stmt)
                    if document_to_update:
                        document_to_update.size = doc.metadata.get("file_size", document_to_update.size)
                        document_to_update.modified_at = doc.metadata.get("modified_at", document_to_update.modified_at)
                        document_to_update.accessed_at = doc.metadata.get("accessed_at", document_to_update.accessed_at)
                        db.commit()
                        print(f"ORM update successful for doc_id: {doc.doc_id}")
                    else:
                        print(f"Document with ID {doc.doc_id} not found.")

            # commit all additions
            db.commit()
        except Exception as e:
            print(f"Error saving metadata: {e}")
            db.rollback()
        finally:
            db.close()

def table_exists_in_db(table_name) -> bool:
    try:
        conn = psycopg2.connect(
            host=url.host, 
            port=url.port,
            user=url.username, 
            password=url.password, 
            database=url.database
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public' AND tablename = '{table_name}'
                );
                """
            )
            return cur.fetchone()[0]
    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False

# delete column in table
def delete_document_metadata_by_doc_id(doc_id: str) -> bool:
    """Delete a document metadata entry by its doc_id."""
    db = SessionLocal()
    try:
        document = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
        if document:
            db.delete(document)
            db.commit()
            print(f"Document metadata with doc_id: {doc_id} deleted.")
            return True
        else:
            print(f"No document metadata found with doc_id: {doc_id}.")
            return False
    except Exception as e:
        print(f"Error deleting document metadata with doc_id: {doc_id}; {e}")
        db.rollback()
        return False
    finally:
        db.close()

