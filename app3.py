
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import os
import hashlib

# Set page configuration
st.set_page_config(page_title="Personal Expense Tracker", layout="wide")

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Initialize user credentials if not exist
if not os.path.exists('credentials.json'):
    # Default credentials (username: admin, password: password)
    default_username = "admin"
    default_password = "Ybok7619"
    
    # Hash the password for storage
    hashed_password = hashlib.sha256(default_password.encode()).hexdigest()
    
    credentials = {
        default_username: hashed_password
    }
    
    with open('credentials.json', 'w') as f:
        json.dump(credentials, f)

# Load credentials
def load_credentials():
    with open('credentials.json', 'r') as f:
        return json.load(f)

# Authentication function
def authenticate(username, password):
    credentials = load_credentials()
    if username in credentials:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_password == credentials[username]
    return False

# Login Page
def login_page():
    st.title("Login to Personal Expense Tracker")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
    


# Initialize session state for data persistence (only when authenticated)
def initialize_app_data():
    if 'accounts' not in st.session_state:
        st.session_state.accounts = {
            "Fedility Account 1": 0,
            "Access Account 2": 0,
            "Mobile Money Account 3": 0,
            "Cash": 0
        }

    if 'transactions' not in st.session_state:
        st.session_state.transactions = []

# Functions for data operations
def save_data():
    """Save current data to files"""
    data = {
        'accounts': st.session_state.accounts,
        'transactions': st.session_state.transactions
    }
    with open('expense_tracker_data.json', 'w') as f:
        json.dump(data, f)

def load_data():
    """Load data from files if they exist"""
    if os.path.exists('expense_tracker_data.json'):
        with open('expense_tracker_data.json', 'r') as f:
            data = json.load(f)
            st.session_state.accounts = data['accounts']
            st.session_state.transactions = data['transactions']

def add_transaction(transaction_type, amount, account, category, description, date):
    """Add a new transaction to the history"""
    transaction = {
        'type': transaction_type,
        'amount': amount,
        'account': account,
        'category': category,
        'description': description,
        'date': date
    }
    
    # Update account balance
    if transaction_type == "Expense":
        st.session_state.accounts[account] -= amount
    else:  # Income
        st.session_state.accounts[account] += amount
    
    # Add to transactions list
    st.session_state.transactions.append(transaction)
    save_data()

# Main application flow
if not st.session_state.authenticated:
    login_page()
else:
    # Initialize and load data after authentication
    initialize_app_data()
    
    # Try to load data at startup
    try:
        load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    # Add logout button to sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # App title and description
    st.title("Personal Expense Tracker")
    st.markdown("Track your expenses, income, and account balances in one place")
    
    # Sidebar for navigation
    page = st.sidebar.radio("Navigation", ["Dashboard", "Add Transaction", "Transaction History", "Account Management"])
    
    # Dashboard page
    if page == "Dashboard":
        st.header("Financial Dashboard")
        
        # Display account balances
        st.subheader("Account Balances")
        account_df = pd.DataFrame({
            'Account': list(st.session_state.accounts.keys()),
            'Balance': list(st.session_state.accounts.values())
        })
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.dataframe(account_df, use_container_width=True)
        
        with col2:
            if sum(st.session_state.accounts.values()) > 0:
                fig = px.pie(account_df, values='Balance', names='Account', title="Balance Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Display recent transactions
        st.subheader("Recent Transactions")
        if st.session_state.transactions:
            recent_transactions = st.session_state.transactions[-5:]
            recent_df = pd.DataFrame(recent_transactions)
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No transactions recorded yet.")
        
        # Display expense summary by category if there are transactions
        if st.session_state.transactions:
            st.subheader("Expense Summary by Category")
            transactions_df = pd.DataFrame(st.session_state.transactions)
            
            # Filter for expenses only
            expenses_df = transactions_df[transactions_df['type'] == 'Expense']
            
            if not expenses_df.empty:
                category_summary = expenses_df.groupby('category')['amount'].sum().reset_index()
                fig = px.bar(category_summary, x='category', y='amount', title="Expenses by Category")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No expense transactions recorded yet.")
    
    # Add Transaction page
    elif page == "Add Transaction":
        st.header("Add New Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            transaction_type = st.selectbox("Transaction Type", ["Expense", "Income"])
            amount = st.number_input("Amount", min_value=0.01, format="%.2f")
            account = st.selectbox("Account", list(st.session_state.accounts.keys()))
            category = st.selectbox("Category", 
                                  ["Food", "Transport", "Housing", "Utilities", "Entertainment", 
                                    "Shopping", "Healthcare", "Education", "Salary", "Gift", "Other"])
            description = st.text_area("Description", placeholder="Enter details about the transaction")
            date = st.date_input("Date", datetime.now())
            
            if st.button("Add Transaction"):
                if amount <= 0:
                    st.error("Amount must be greater than zero")
                elif transaction_type == "Expense" and amount > st.session_state.accounts[account]:
                    st.error(f"Insufficient funds in {account}")
                else:
                    add_transaction(transaction_type, amount, account, category, description, date.strftime("%Y-%m-%d"))
                    st.success(f"{transaction_type} of ${amount:.2f} added successfully!")
        
        with col2:
            st.subheader("Current Account Balances")
            for account, balance in st.session_state.accounts.items():
                st.metric(account, f"${balance:.2f}")
    
    # Transaction History page
    elif page == "Transaction History":
        st.header("Transaction History")
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.multiselect("Transaction Type", ["Expense", "Income"], default=["Expense", "Income"])
        
        with col2:
            filter_account = st.multiselect("Account", list(st.session_state.accounts.keys()), 
                                          default=list(st.session_state.accounts.keys()))
        
        with col3:
            filter_category = st.multiselect("Category", 
                                           ["Food", "Transport", "Housing", "Utilities", "Entertainment", 
                                            "Shopping", "Healthcare", "Education", "Salary", "Gift", "Other"],
                                           default=["Food", "Transport", "Housing", "Utilities", "Entertainment", 
                                                   "Shopping", "Healthcare", "Education", "Salary", "Gift", "Other"])
        
        # Display filtered transactions
        if st.session_state.transactions:
            transactions_df = pd.DataFrame(st.session_state.transactions)
            
            # Apply filters
            filtered_df = transactions_df[
                (transactions_df['type'].isin(filter_type)) & 
                (transactions_df['account'].isin(filter_account)) & 
                (transactions_df['category'].isin(filter_category))
            ]
            
            if not filtered_df.empty:
                st.dataframe(filtered_df.sort_values('date', ascending=False), use_container_width=True)
                
                # Download option
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Transaction History",
                    csv,
                    "transaction_history.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.info("No transactions match your filter criteria.")
        else:
            st.info("No transactions recorded yet.")
    
    # Account Management page
    elif page == "Account Management":
        st.header("Account Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Update Account Balances")
            account_to_update = st.selectbox("Select Account", list(st.session_state.accounts.keys()))
            current_balance = st.session_state.accounts[account_to_update]
            st.info(f"Current Balance: ${current_balance:.2f}")
            new_balance = st.number_input("New Balance", min_value=0.0, value=float(current_balance), format="%.2f")
            
            if st.button("Update Balance"):
                st.session_state.accounts[account_to_update] = new_balance
                save_data()
                st.success(f"{account_to_update} balance updated to ${new_balance:.2f}")
        
        with col2:
            st.subheader("Account Balances")
            account_data = pd.DataFrame({
                'Account': list(st.session_state.accounts.keys()),
                'Balance': list(st.session_state.accounts.values())
            })
            st.dataframe(account_data, use_container_width=True)
            
            total_balance = sum(st.session_state.accounts.values())
            st.metric("Total Balance", f"${total_balance:.2f}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Personal Expense Tracker - v1.0")
