"""
Attachment Handler - Processes and summarizes WhatsApp attachments
Handles images, voice notes, documents and other media types
"""

import logging
import os
import re
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import base64
import io

# Try to import optional dependencies
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)

class AttachmentHandler:
    """Handles processing of various attachment types in WhatsApp messages"""
    
    def __init__(self, whatsapp_client=None):
        self.whatsapp_client = whatsapp_client
        self.supported_types = {
            'image': ['image/jpeg', 'image/png', 'image/gif'],
            'audio': ['audio/ogg', 'audio/mp3', 'audio/wav', 'audio/mpeg'],
            'document': ['application/pdf', 'application/msword', 
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'text/plain', 'application/vnd.ms-excel',
                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
            'video': ['video/mp4', 'video/3gpp']
        }
    
    async def process_attachment(self, attachment_data: Dict[str, Any], sender: str) -> Dict[str, Any]:
        """
        Process an attachment and return a summary
        
        Args:
            attachment_data: Dictionary containing attachment information
            sender: Phone number of the sender
            
        Returns:
            Dictionary with processed attachment information and summary
        """
        try:
            attachment_type = attachment_data.get('type', 'unknown')
            mime_type = attachment_data.get('mime_type', '')
            attachment_id = attachment_data.get('id')
            
            logger.info(f"Processing {attachment_type} attachment from {sender}")
            
            # Process based on attachment type
            if attachment_type == 'image':
                return await self._process_image(attachment_data, sender)
            elif attachment_type == 'audio':
                return await self._process_audio(attachment_data, sender)
            elif attachment_type == 'document':
                return await self._process_document(attachment_data, sender)
            elif attachment_type == 'video':
                return await self._process_video(attachment_data, sender)
            else:
                return {
                    'type': attachment_type,
                    'mime_type': mime_type,
                    'id': attachment_id,
                    'summary': f"Unsupported attachment type: {attachment_type}",
                    'processed_at': datetime.utcnow().isoformat(),
                    'sender': sender
                }
                
        except Exception as e:
            logger.error(f"Error processing attachment from {sender}: {e}")
            return {
                'type': attachment_data.get('type', 'unknown'),
                'mime_type': attachment_data.get('mime_type', ''),
                'id': attachment_data.get('id'),
                'summary': f"Error processing attachment: {str(e)}",
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'error': str(e)
            }
    
    async def _process_image(self, attachment_data: Dict[str, Any], sender: str) -> Dict[str, Any]:
        """Process image attachments"""
        try:
            attachment_id = attachment_data.get('id')
            mime_type = attachment_data.get('mime_type')
            
            # For now, we'll create a basic summary since we don't have actual image processing
            # In a full implementation, we would:
            # 1. Download the image using the WhatsApp client
            # 2. Use OCR to extract text
            # 3. Use computer vision to describe the image
            
            summary = f"Image attachment ({mime_type})"
            
            # If we had PIL available, we could get image dimensions
            if PIL_AVAILABLE:
                summary += " - Image processing capabilities available"
            else:
                summary += " - Basic image handling (install Pillow for advanced features)"
            
            return {
                'type': 'image',
                'mime_type': mime_type,
                'id': attachment_id,
                'summary': summary,
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'image_info': {
                    'description': 'Image received - content analysis requires advanced processing',
                    'contains_text': 'Unknown - OCR not performed',
                    'objects': 'Unknown - Computer vision not performed'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image from {sender}: {e}")
            return {
                'type': 'image',
                'mime_type': attachment_data.get('mime_type', ''),
                'id': attachment_data.get('id'),
                'summary': f"Error processing image: {str(e)}",
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'error': str(e)
            }
    
    async def _process_audio(self, attachment_data: Dict[str, Any], sender: str) -> Dict[str, Any]:
        """Process audio attachments (voice notes)"""
        try:
            attachment_id = attachment_data.get('id')
            mime_type = attachment_data.get('mime_type')
            
            # For voice notes, we would typically:
            # 1. Download the audio file
            # 2. Convert to a compatible format if needed
            # 3. Use speech recognition to transcribe
            
            summary = f"Audio attachment ({mime_type})"
            
            if SPEECH_RECOGNITION_AVAILABLE:
                summary += " - Speech recognition available"
                transcription_status = "Ready for transcription (implementation required)"
            else:
                summary += " - Basic audio handling (install SpeechRecognition for transcription)"
                transcription_status = "Speech recognition not available"
            
            return {
                'type': 'audio',
                'mime_type': mime_type,
                'id': attachment_id,
                'summary': summary,
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'audio_info': {
                    'duration': 'Unknown - Audio analysis not performed',
                    'transcription': transcription_status,
                    'language': 'Unknown'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing audio from {sender}: {e}")
            return {
                'type': 'audio',
                'mime_type': attachment_data.get('mime_type', ''),
                'id': attachment_data.get('id'),
                'summary': f"Error processing audio: {str(e)}",
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'error': str(e)
            }
    
    async def _process_document(self, attachment_data: Dict[str, Any], sender: str) -> Dict[str, Any]:
        """Process document attachments"""
        try:
            attachment_id = attachment_data.get('id')
            mime_type = attachment_data.get('mime_type')
            
            # For documents, we would typically:
            # 1. Download the document
            # 2. Extract text content
            # 3. Summarize the content
            
            # Determine document type for better description
            if 'pdf' in mime_type:
                doc_type = "PDF document"
            elif 'word' in mime_type or 'document' in mime_type:
                doc_type = "Word document"
            elif 'excel' in mime_type or 'spreadsheet' in mime_type:
                doc_type = "Excel spreadsheet"
            elif 'text' in mime_type or 'plain' in mime_type:
                doc_type = "Text file"
            else:
                doc_type = "Document"
            
            summary = f"{doc_type} ({mime_type})"
            
            return {
                'type': 'document',
                'mime_type': mime_type,
                'id': attachment_id,
                'summary': summary,
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'document_info': {
                    'type': doc_type,
                    'content_summary': 'Document content analysis requires processing',
                    'page_count': 'Unknown',
                    'contains': 'Unknown - Document analysis not performed'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document from {sender}: {e}")
            return {
                'type': 'document',
                'mime_type': attachment_data.get('mime_type', ''),
                'id': attachment_data.get('id'),
                'summary': f"Error processing document: {str(e)}",
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'error': str(e)
            }
    
    async def _process_video(self, attachment_data: Dict[str, Any], sender: str) -> Dict[str, Any]:
        """Process video attachments"""
        try:
            attachment_id = attachment_data.get('id')
            mime_type = attachment_data.get('mime_type')
            
            # For videos, we would typically:
            # 1. Download the video
            # 2. Extract key frames
            # 3. Use computer vision to analyze content
            # 4. Extract audio and transcribe
            
            summary = f"Video attachment ({mime_type})"
            
            return {
                'type': 'video',
                'mime_type': mime_type,
                'id': attachment_id,
                'summary': summary,
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'video_info': {
                    'duration': 'Unknown - Video analysis not performed',
                    'content_description': 'Video content analysis requires advanced processing',
                    'contains_audio': 'Unknown',
                    'transcription': 'Video transcription requires processing'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing video from {sender}: {e}")
            return {
                'type': 'video',
                'mime_type': attachment_data.get('mime_type', ''),
                'id': attachment_data.get('id'),
                'summary': f"Error processing video: {str(e)}",
                'processed_at': datetime.utcnow().isoformat(),
                'sender': sender,
                'error': str(e)
            }
    
    def get_attachment_summary(self, attachment_info: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of an attachment for inclusion in conversation context
        
        Args:
            attachment_info: Processed attachment information
            
        Returns:
            String summary of the attachment
        """
        try:
            attachment_type = attachment_info.get('type', 'unknown')
            summary = attachment_info.get('summary', '')
            
            # Create a more descriptive summary based on attachment type
            if attachment_type == 'image':
                return f"[Image attachment: {summary}]"
            elif attachment_type == 'audio':
                audio_info = attachment_info.get('audio_info', {})
                transcription_status = audio_info.get('transcription', 'Audio processing not performed')
                return f"[Audio attachment: {summary} - {transcription_status}]"
            elif attachment_type == 'document':
                document_info = attachment_info.get('document_info', {})
                doc_type = document_info.get('type', 'Document')
                return f"[{doc_type}: {summary}]"
            elif attachment_type == 'video':
                video_info = attachment_info.get('video_info', {})
                content_desc = video_info.get('content_description', 'Video content')
                return f"[Video attachment: {summary} - {content_desc}]"
            else:
                return f"[Attachment: {summary}]"
                
        except Exception as e:
            logger.error(f"Error creating attachment summary: {e}")
            return "[Attachment: Processing error]"
    
    async def transcribe_audio(self, audio_data: bytes, mime_type: str) -> str:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio data
            mime_type: MIME type of the audio
            
        Returns:
            Transcribed text
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            return "Speech recognition not available. Install SpeechRecognition package for transcription."
        
        try:
            # This is a placeholder implementation
            # In a full implementation, we would:
            # 1. Convert audio to a format supported by speech recognition
            # 2. Use speech recognition library to transcribe
            # 3. Return the transcription
            
            return "Audio transcription would be available here with proper implementation"
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    async def summarize_document(self, document_data: bytes, mime_type: str) -> str:
        """
        Summarize document content
        
        Args:
            document_data: Raw document data
            mime_type: MIME type of the document
            
        Returns:
            Document summary
        """
        try:
            # This is a placeholder implementation
            # In a full implementation, we would:
            # 1. Extract text from the document based on type
            # 2. Use AI to summarize the content
            # 3. Return the summary
            
            return "Document summary would be available here with proper implementation"
            
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return f"Error summarizing document: {str(e)}"

# Global attachment handler instance
attachment_handler = AttachmentHandler()