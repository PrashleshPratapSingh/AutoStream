"""
Tool definitions for the AutoStream AI Agent.
Contains the mock lead capture function as required by the assignment.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API function that simulates capturing a lead.
    
    In production, this would send data to a CRM or database.
    
    Args:
        name: The lead's full name.
        email: The lead's email address.
        platform: The creator's primary platform (YouTube, Instagram, etc.)
    
    Returns:
        Confirmation message string.
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead captured successfully: {name}, {email}, {platform}"
