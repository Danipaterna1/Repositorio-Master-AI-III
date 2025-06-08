"""
Smart Semantic Chunker - Optimized for Large Volume Processing

Motor de chunking inteligente que preserva contexto y optimiza para retrieval.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Lazy imports to avoid dependency conflicts
nltk = None
SentenceTransformer = None
np = None
cosine_similarity = None

def _load_dependencies():
    """Load dependencies only when needed"""
    global nltk, SentenceTransformer, np, cosine_similarity
    
    if nltk is None:
        try:
            import nltk as _nltk
            nltk = _nltk
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
        except ImportError:
            print("Warning: NLTK not available, using basic text processing")
            
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except ImportError:
            print("Warning: SentenceTransformers not available")
            
    if np is None:
        try:
            import numpy as _np
            np = _np
        except ImportError:
            print("Warning: NumPy not available")
            
    if cosine_similarity is None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity as _cs
            cosine_similarity = _cs
        except ImportError:
            print("Warning: Scikit-learn not available")

class ChunkType(str, Enum):
    """Tipos de chunks por estructura semántica"""
    PARAGRAPH = "paragraph"
    SECTION = "section" 
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    HEADER = "header"

@dataclass
class SmartChunk:
    """Chunk con metadata semántica rica"""
    id: str
    content: str
    chunk_type: ChunkType
    position: int
    word_count: int
    char_count: int
    
    # Context preservation
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    parent_section: Optional[str] = None
    
    # Semantic metadata
    key_phrases: List[str] = None
    entities: List[str] = None
    topics: List[str] = None
    complexity_score: float = 0.0
    
    # Document metadata
    document_id: str = ""
    document_title: str = ""
    page_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "position": self.position,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "parent_section": self.parent_section,
            "key_phrases": self.key_phrases or [],
            "entities": self.entities or [],
            "topics": self.topics or [],
            "complexity_score": self.complexity_score,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "page_number": self.page_number
        }

class SmartSemanticChunker:
    """
    Chunker semántico optimizado para performance y calidad de contexto.
    
    Features:
    - Chunking adaptativo según tipo de contenido
    - Preservación de contexto entre chunks
    - Metadata semántica rica
    - Optimizado para batch processing
    """
    
    def __init__(self, 
                 target_chunk_size: int = 512,
                 max_chunk_size: int = 1024,
                 min_chunk_size: int = 100,
                 overlap_size: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
        # Load dependencies when initializing
        _load_dependencies()
        
        # Load embedding model for semantic similarity (if available)
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(f"sentence-transformers/{embedding_model}")
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # Compiled regex patterns for performance
        self.patterns = {
            'headers': re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),
            'code_blocks': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'lists': re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE),
            'tables': re.compile(r'^\|(.+\|)+$', re.MULTILINE),
            'paragraphs': re.compile(r'\n\s*\n'),
            'sentences': re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        }
    
    def chunk_document(self, 
                      text: str, 
                      document_id: str,
                      document_title: str = "",
                      preserve_structure: bool = True) -> List[SmartChunk]:
        """
        Chunk a document with intelligent semantic preservation.
        
        Args:
            text: Document text
            document_id: Unique document identifier
            document_title: Document title for metadata
            preserve_structure: Whether to preserve document structure
            
        Returns:
            List of SmartChunk objects with rich metadata
        """
        
        # Preprocessing
        text = self._preprocess_text(text)
        
        if preserve_structure:
            chunks = self._structure_aware_chunking(text, document_id, document_title)
        else:
            chunks = self._simple_semantic_chunking(text, document_id, document_title)
        
        # Post-processing: add connections and metadata
        chunks = self._add_chunk_connections(chunks)
        chunks = self._extract_semantic_metadata(chunks)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _structure_aware_chunking(self, 
                                 text: str, 
                                 document_id: str, 
                                 document_title: str) -> List[SmartChunk]:
        """
        Intelligent chunking that preserves document structure.
        """
        chunks = []
        current_section = None
        position = 0
        
        # Split by major structural elements
        sections = self._identify_sections(text)
        
        for section_title, section_content in sections:
            current_section = section_title
            
            # Handle different content types
            if self._is_code_block(section_content):
                chunk = self._create_code_chunk(section_content, position, document_id, document_title, current_section)
                chunks.append(chunk)
                position += 1
                
            elif self._is_table(section_content):
                chunk = self._create_table_chunk(section_content, position, document_id, document_title, current_section)
                chunks.append(chunk)
                position += 1
                
            else:
                # Regular text - use semantic chunking
                text_chunks = self._semantic_text_chunking(section_content, position, document_id, document_title, current_section)
                chunks.extend(text_chunks)
                position += len(text_chunks)
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify major sections in the document"""
        sections = []
        
        # Find headers
        header_matches = list(self.patterns['headers'].finditer(text))
        
        if not header_matches:
            # No headers found, treat as single section
            return [("Main Content", text)]
        
        last_end = 0
        for i, match in enumerate(header_matches):
            # Add content before first header
            if i == 0 and match.start() > 0:
                sections.append(("Introduction", text[last_end:match.start()].strip()))
            
            # Determine section end
            next_start = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
            
            section_title = match.group(1)
            section_content = text[match.end():next_start].strip()
            
            if section_content:  # Only add non-empty sections
                sections.append((section_title, section_content))
        
        return sections
    
    def _semantic_text_chunking(self, 
                               text: str, 
                               start_position: int,
                               document_id: str,
                               document_title: str,
                               section_title: str) -> List[SmartChunk]:
        """
        Semantic chunking for regular text content.
        """
        chunks = []
        
        # Use nltk if available, otherwise simple sentence splitting
        if nltk:
            try:
                sentences = nltk.sent_tokenize(text)
            except:
                # Fallback to simple splitting
                sentences = self._simple_sentence_split(text)
        else:
            sentences = self._simple_sentence_split(text)
        
        current_chunk = ""
        current_sentences = []
        position = start_position
        
        for sentence in sentences:
            # Check if adding sentence would exceed target size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk.split()) > self.target_chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_text_chunk(
                    current_chunk, 
                    position, 
                    document_id, 
                    document_title, 
                    section_title
                )
                chunks.append(chunk)
                position += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
                
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
        
        # Add remaining content as final chunk
        if current_chunk.strip():
            chunk = self._create_text_chunk(
                current_chunk, 
                position, 
                document_id, 
                document_title, 
                section_title
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_text_chunk(self, 
                          content: str, 
                          position: int,
                          document_id: str,
                          document_title: str,
                          section_title: str) -> SmartChunk:
        """Create a text chunk with metadata"""
        
        chunk_id = self._generate_chunk_id(document_id, position, content)
        word_count = len(content.split())
        char_count = len(content)
        
        return SmartChunk(
            id=chunk_id,
            content=content.strip(),
            chunk_type=ChunkType.PARAGRAPH,
            position=position,
            word_count=word_count,
            char_count=char_count,
            parent_section=section_title,
            document_id=document_id,
            document_title=document_title
        )
    
    def _create_code_chunk(self, 
                          content: str, 
                          position: int,
                          document_id: str,
                          document_title: str,
                          section_title: str) -> SmartChunk:
        """Create a code block chunk"""
        
        chunk_id = self._generate_chunk_id(document_id, position, content)
        
        return SmartChunk(
            id=chunk_id,
            content=content.strip(),
            chunk_type=ChunkType.CODE_BLOCK,
            position=position,
            word_count=len(content.split()),
            char_count=len(content),
            parent_section=section_title,
            document_id=document_id,
            document_title=document_title,
            complexity_score=0.8  # Code is typically complex
        )
    
    def _create_table_chunk(self, 
                           content: str, 
                           position: int,
                           document_id: str,
                           document_title: str,
                           section_title: str) -> SmartChunk:
        """Create a table chunk"""
        
        chunk_id = self._generate_chunk_id(document_id, position, content)
        
        return SmartChunk(
            id=chunk_id,
            content=content.strip(),
            chunk_type=ChunkType.TABLE,
            position=position,
            word_count=len(content.split()),
            char_count=len(content),
            parent_section=section_title,
            document_id=document_id,
            document_title=document_title,
            complexity_score=0.6  # Tables are moderately complex
        )
    
    def _add_chunk_connections(self, chunks: List[SmartChunk]) -> List[SmartChunk]:
        """Add previous/next chunk connections"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.previous_chunk_id = chunks[i-1].id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i+1].id
        
        return chunks
    
    def _extract_semantic_metadata(self, chunks: List[SmartChunk]) -> List[SmartChunk]:
        """Extract semantic metadata for each chunk"""
        
        # Batch processing for efficiency
        contents = [chunk.content for chunk in chunks]
        
        # Extract key phrases (simplified)
        for chunk in chunks:
            chunk.key_phrases = self._extract_key_phrases(chunk.content)
            chunk.complexity_score = self._calculate_complexity(chunk.content)
        
        return chunks
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text (simplified implementation)"""
        # Simple extraction based on word frequency and length
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top frequent words as key phrases
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5] if freq > 1]
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback when nltk is not available"""
        # Basic sentence splitting using common punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        # Clean empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        words = text.split()
        
        # Use nltk if available, otherwise simple splitting
        if nltk:
            try:
                sentences = nltk.sent_tokenize(text)
            except:
                sentences = self._simple_sentence_split(text)
        else:
            sentences = self._simple_sentence_split(text)
        
        if not sentences:
            return 0.0
        
        # Average words per sentence (higher = more complex)
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Long words ratio (6+ characters)
        long_words = [w for w in words if len(w) >= 6]
        long_word_ratio = len(long_words) / len(words) if words else 0
        
        # Combine metrics (normalized to 0-1)
        complexity = min(1.0, (avg_words_per_sentence / 20) * 0.5 + long_word_ratio * 0.5)
        
        return round(complexity, 2)
    
    def _generate_chunk_id(self, document_id: str, position: int, content: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{document_id}_chunk_{position:04d}_{content_hash}"
    
    def _is_code_block(self, text: str) -> bool:
        """Check if text is a code block"""
        return bool(self.patterns['code_blocks'].search(text))
    
    def _is_table(self, text: str) -> bool:
        """Check if text contains tables"""
        lines = text.split('\n')
        table_lines = [line for line in lines if self.patterns['tables'].match(line.strip())]
        return len(table_lines) >= 2  # At least 2 table rows
    
    def _simple_semantic_chunking(self, 
                                 text: str, 
                                 document_id: str, 
                                 document_title: str) -> List[SmartChunk]:
        """Simple semantic chunking when structure preservation is disabled"""
        return self._semantic_text_chunking(text, 0, document_id, document_title, "Main Content")

# Factory function
def create_smart_chunker(config: Dict[str, Any] = None) -> SmartSemanticChunker:
    """Create chunker with configuration"""
    config = config or {}
    
    return SmartSemanticChunker(
        target_chunk_size=config.get('target_chunk_size', 512),
        max_chunk_size=config.get('max_chunk_size', 1024),
        min_chunk_size=config.get('min_chunk_size', 100),
        overlap_size=config.get('overlap_size', 50),
        embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2')
    ) 