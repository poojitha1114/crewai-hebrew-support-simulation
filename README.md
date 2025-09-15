# Hebrew Customer Service Call Simulation

A complete CrewAI project that simulates Hebrew customer cancellation calls using multiple AI agents for realistic conversation flow.

## Overview

This project implements a sophisticated customer service simulation where a Hebrew-speaking customer attempts to cancel their TV subscription while interacting with a customer service representative. The system uses multiple specialized agents to handle different aspects of the conversation:

- **NikudAgent**: Processes Hebrew text with vowel points (nikud) for accurate pronunciation
- **TTSAgent**: Converts Hebrew text to speech using Chatterbox
- **STTAgent**: Transcribes Hebrew speech to text using Whisper
- **ClientAgent**: Simulates the customer with realistic personality and responses
- **CSRAgent**: Acts as customer service representative using LiteLLM
- **TranscriptAgent**: Maintains detailed conversation logs
- **TokenMonitorAgent**: Manages LLM token usage and triggers summarization
- **GuardrailAgent**: Enforces safety limits and PII protection

## Features

- ✅ Complete Hebrew language support with nikud processing
- ✅ Realistic conversation flow with personality-driven responses
- ✅ Token budget management with automatic summarization
- ✅ PII detection and sanitization
- ✅ Comprehensive logging and transcription
- ✅ Audio file generation and processing
- ✅ Fallback mechanisms for service failures
- ✅ Configurable guardrails and safety limits

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ClientAgent   │    │    CSRAgent     │    │ TranscriptAgent │
│                 │    │                 │    │                 │
│ • Hebrew phrases│    │ • LiteLLM       │    │ • JSON logs     │
│ • Personality   │    │ • Retention     │    │ • Timestamps    │
│ • State mgmt    │    │ • Escalation    │    │ • Audio paths   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Main Flow     │
                    │                 │
                    │ • Orchestration │
                    │ • Turn mgmt     │
                    │ • Error handling│
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NikudAgent    │    │   TTSAgent      │    │   STTAgent      │
│                 │    │                 │    │                 │
│ • Phonikud API  │    │ • Chatterbox    │    │ • Whisper       │
│ • Text processing│    │ • Audio gen     │    │ • Transcription │
│ • Confidence    │    │ • File mgmt     │    │ • Confidence    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### External Services

You need to set up the following services:

1. **Phonikud Server** - Hebrew nikud processing
2. **Chatterbox Server** - Hebrew text-to-speech
3. **Whisper Server** (optional) - Speech-to-text (can use local Whisper)

### System Requirements

- Python 3.11+
- 4GB+ RAM (for Whisper models)
- Audio processing libraries
- Internet connection for LLM APIs

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd crewai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Service URLs
PHONIKUD_URL=http://localhost:8000
CHATTERBOX_URL=http://localhost:8001
WHISPER_URL=http://localhost:8002
WHISPER_LOCAL=true
WHISPER_MODEL=base

# LLM Configuration
PRIMARY_MODEL=gpt-4o-mini
FALLBACK_MODELS=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=500

# OpenAI API Key (required for LiteLLM)
OPENAI_API_KEY=your_openai_api_key_here

# Conversation Settings
MAX_TURNS=15
MAX_CONVERSATION_TOKENS=10000
CLIENT_PERSONALITY=polite_but_determined
OUTPUT_DIR=output
```

## External Service Setup

### 1. Phonikud Server

```bash
# Clone Phonikud
git clone https://github.com/thewh1teagle/phonikud.git
cd phonikud

# Install and run
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Chatterbox Server

```bash
# Clone Chatterbox
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox

# Follow their installation instructions
# Run server on port 8001
python server.py --port 8001
```

### 3. Whisper Setup (Local)

Whisper will be installed automatically with the requirements. For better performance, you can install with GPU support:

```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Quick Start Demo

```bash
python main.py
```

This runs a complete conversation simulation with default settings and automatically generates merged audio with distinct male/female voices.

### Listen to Generated Audio

After running the conversation, you can listen to the audio in several ways:

#### Option 1: Play Complete Merged Conversation (Recommended)
```bash
python play_conversation_audio.py
```
This plays the complete conversation with both Hebrew male voice (client) and Hebrew female voice (CSR), including audio markers.

#### Option 2: Open Audio File Directly
```bash
# Windows
start output\conversation_complete_conv_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.wav

# macOS
open output/conversation_complete_conv_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.wav

# Linux
xdg-open output/conversation_complete_conv_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.wav
```

#### Option 3: Manual Audio Merging
```bash
# Merge latest conversation
python merge_conversation_audio.py

# Merge specific conversation by ID
python merge_conversation_audio.py conv_20241213_143022
```

### Audio Features
- **Male Voice**: Hebrew client (customer) - `he-IL-AvriNeural`
- **Female Voice**: Hebrew CSR (support agent) - `he-IL-HilaNeural`
- **Audio Markers**: 
  - Single beep = Client speaking
  - Double beep = CSR speaking
- **Complete Conversations**: All turns merged into one audio file

### Custom Configuration

```python
from main import HebrewCustomerServiceFlow, load_configuration

# Load and modify configuration
config = load_configuration()
config['max_turns'] = 10
config['client_personality'] = 'frustrated'

# Create and run flow
flow = HebrewCustomerServiceFlow(config)
result = flow.run_full_conversation()

print(f"Conversation outcome: {result['outcome']}")
print(f"Transcript saved: {result['transcript_path']}")
```

### Individual Agent Usage

```python
from agents.nikud_agent import NikudAgent
from agents.tts_agent import TTSAgent

# Process Hebrew text
nikud_agent = NikudAgent()
result = nikud_agent.add_nikud("אני רוצה לבטל את המנוי")
print(f"With nikud: {result.nikud_text}")

# Generate speech
tts_agent = TTSAgent()
audio_result = tts_agent.synthesize_speech(
    result.nikud_text, 
    voice="hebrew_female",
    speaker="client",
    turn=1
)
print(f"Audio saved: {audio_result.audio_file_path}")
```

## Output Files

The system generates several output files in the `output/` directory:

### Audio Files
- `client_turn_001_YYYYMMDD_HHMMSS.wav` - Client speech audio
- `csr_turn_002_YYYYMMDD_HHMMSS.wav` - CSR speech audio (if enabled)

### Transcripts
- `transcript_conv_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.json` - Complete conversation log

### Logs
- `conversation.log` - Detailed system logs

### Example Transcript Structure

```json
{
  "metadata": {
    "conversation_id": "conv_20241213_143022",
    "start_time": "2024-12-13T14:30:22",
    "end_time": "2024-12-13T14:35:45",
    "total_turns": 8,
    "client_personality": "polite_but_determined",
    "outcome": "cancellation_approved",
    "total_tokens_used": 2847
  },
  "entries": [
    {
      "timestamp": "2024-12-13T14:30:22",
      "turn": 1,
      "speaker": "client",
      "text": "שלום, אני רוצה לבטל את המנוי שלי לטלוויזיה",
      "nikud_text": "שָׁלוֹם, אֲנִי רוֹצֶה לְבַטֵּל אֶת הַמְּנוֹי שֶׁלִּי לַטֶּלֶוִיזְיָה",
      "audio_file_path": "output/client_turn_001_20241213_143022.wav",
      "confidence": 1.0,
      "duration": 3.2
    }
  ]
}
```

## Configuration Options

### Client Personalities
- `polite_but_determined` - Polite but persistent
- `frustrated` - Shows increasing frustration
- `business_like` - Direct and professional
- `confused` - Needs more explanation

### LLM Models
- Primary: `gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`
- Fallback: Automatic fallback to secondary models

### Whisper Models
- `tiny` - Fastest, least accurate
- `base` - Good balance (recommended)
- `small` - Better accuracy
- `medium` - High accuracy
- `large` - Best accuracy, slowest

## Troubleshooting

### Common Issues

1. **Service Connection Errors**
   ```bash
   # Check if services are running
   curl http://localhost:8000/health  # Phonikud
   curl http://localhost:8001/health  # Chatterbox
   ```

2. **Audio Processing Issues**
   ```bash
   # Install additional audio libraries
   pip install ffmpeg-python
   # On Ubuntu: sudo apt-get install ffmpeg
   # On macOS: brew install ffmpeg
   ```

3. **Memory Issues with Whisper**
   ```python
   # Use smaller model
   config['whisper_model'] = 'tiny'
   ```

4. **Token Limit Exceeded**
   ```python
   # Reduce max tokens or enable more aggressive summarization
   config['max_tokens'] = 5000
   config['llm_config']['max_tokens'] = 300
   ```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/ -v
```

### Unit Tests

```bash
# Test individual agents
pytest tests/test_nikud_agent.py
pytest tests/test_tts_agent.py
pytest tests/test_client_agent.py
```

## Performance Optimization

### For Production Use

1. **Use GPU for Whisper**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Optimize Token Usage**
   ```python
   config['llm_config']['max_tokens'] = 300  # Reduce response length
   config['max_conversation_tokens'] = 5000  # Trigger summarization earlier
   ```

3. **Cache Nikud Results**
   ```python
   # Implement caching in NikudAgent for repeated phrases
   ```

4. **Batch Processing**
   ```python
   # Use batch methods for multiple conversations
   tts_agent.batch_synthesize(texts)
   stt_agent.batch_transcribe(audio_files)
   ```

## API Reference

### Main Flow Class

```python
class HebrewCustomerServiceFlow:
    def __init__(self, config: Dict)
    def start_conversation(self) -> str
    def conversation_turn(self, csr_message: str = "") -> Dict
    def run_full_conversation(self) -> Dict
```

### Agent Interfaces

All agents follow the CrewAI pattern:
```python
def create_task(self, **kwargs) -> Task
def execute_task(self, task: Task) -> ResponseModel
def get_stats(self) -> Dict
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent framework
- [Phonikud](https://github.com/thewh1teagle/phonikud) - Hebrew nikud processing
- [Chatterbox](https://github.com/resemble-ai/chatterbox) - Text-to-speech
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM integration

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `output/conversation.log`
3. Open an issue on GitHub with:
   - Error messages
   - Configuration used
   - Steps to reproduce

---

**Note**: This is a simulation system for educational and testing purposes. Ensure you have proper permissions and comply with privacy regulations when processing real customer data.
"# crewai-hebrew-support-simulation" 
