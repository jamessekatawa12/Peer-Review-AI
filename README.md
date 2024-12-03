# AI Multi-Agent Framework for Peer Review

An open-source framework that revolutionizes the academic peer review process through specialized AI agents, fostering interdisciplinary collaboration and ensuring rigorous review standards.

## Features

- **Multi-Agent System**
  - Plagiarism Detection Agent
  - Ethics Review Agent
  - Methodology Analysis Agent
  - Citation Verification Agent

- **Knowledge Graph**
  - Cross-domain concept mapping
  - Interdisciplinary connection discovery
  - Visual concept exploration
  - Discipline overlap analysis

- **Modern Architecture**
  - FastAPI backend
  - React frontend
  - Async agent coordination
  - Modular design

## Prerequisites

- Python 3.8+
- Node.js 14+
- PyTorch
- Hugging Face Transformers
- MongoDB (optional, for persistence)

## Installation

1. Clone the repository

```bash
git clone https://github.com/jamessekatawa12/peer_review_ai.git
cd peer_review_ai
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Install UI dependencies

```bash
cd ui
npm install
```

## Running the Application

1. Start the backend server

```bash
cd peer_review_ai
uvicorn api.main:app --reload
```

2. Start the frontend development server

```bash
cd ui
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Endpoints

### Manuscript Review
- `POST /api/v1/submit` - Submit a manuscript for review
- `GET /api/v1/status/{manuscript_id}` - Check review status
- `POST /api/v1/upload-pdf` - Upload PDF manuscript

### Knowledge Graph
- `GET /api/v1/knowledge-graph/concepts/{concept}` - Get concept connections
- `GET /api/v1/knowledge-graph/disciplines/{discipline1}/{discipline2}` - Get discipline overlap
- `GET /api/v1/knowledge-graph/central-concepts` - Get central concepts

### System
- `GET /api/v1/agents` - List available agents and capabilities

## Project Structure

```
peer_review_ai/
├── agents/             # AI Agents
│   ├── base/          # Base agent classes
│   ├── disciplinary/  # Discipline-specific agents
│   └── ethics/       # Ethics review agent
├── api/              # FastAPI backend
├── core/             # Core framework components
├── ui/               # React frontend
├── tests/            # Test suites
└── docs/             # Documentation
```

## Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement required methods:
   - `initialize()`
   - `review()`
   - `collaborate()`
3. Register the agent in `api/main.py`

Example:

```python
from agents.base.agent import BaseAgent, AgentType, ReviewResult

class YourAgent(BaseAgent):
    def __init__(self, name: str = "Your Agent"):
        super().__init__(name, AgentType.DISCIPLINARY)
    
    async def initialize(self) -> None:
        # Initialize your agent
        pass
    
    async def review(self, content: str, metadata: Dict[str, Any]) -> ReviewResult:
        # Implement review logic
        pass
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and development process.

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Academic institutions and journals supporting this initiative
- Open-source AI community
- Contributors and maintainers

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{peer_review_ai2024,
  title = {AI Multi-Agent Framework for Peer Review},
  author = {James Sekatawa},
  year = {2024},
  url = {https://github.com/jamessekatawa12/peer_review_ai}
}
```
