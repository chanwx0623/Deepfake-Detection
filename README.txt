1. Setup and Installation
To avoid conflicts with your system Python and other projects, we recommend creating a virtual environment

For Windows:
- python -m venv venv
- venv\Scripts\activate

For macOS/Linux
- python3 -m venv venv
- source venv/bin/activate

2. Install Libraries
With the virtual environment active, install all necessary libraries from the requirements.txt file
- pip install -r requirements.txt

3.Run the Application
After installing the libraries, you can run the application with Streamlit
- streamlit run interface.py
