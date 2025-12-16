"""
Chatbox helper module for RNA structure analysis using RCSB PDB data and Azure OpenAI.
"""

import requests
from typing import Dict, List, Optional, Any


def fetch_rcsb_data(pdb_id: str) -> Dict[str, Any]:
    """
    Fetch comprehensive PDB information from RCSB PDB API.
    
    Args:
        pdb_id: PDB identifier (e.g., "2GDI")
        
    Returns:
        Dictionary containing structure information
    """
    data = {
        'pdb_id': pdb_id,
        'title': None,
        'description': None,
        'experimental_method': None,
        'resolution': None,
        'authors': [],
        'citation': None,
        'doi': None,
        'pubmed_id': None,
        'pubmed_abstract': None,
        'ligands': [],  # Simple list of ligand IDs like ["K", "MG", "TPP"]
        'mesh_terms': []
    }
    
    entry_data = {}
    
    try:
        # Fetch main entry data
        api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            try:
                entry_data = response.json()
            except Exception:
                entry_data = {}
        
        # Basic structure information
        data['title'] = entry_data.get('struct', {}).get('title', 'Title not available')
        data['description'] = entry_data.get('struct', {}).get('pdbx_descriptor', '')
        
        # Experimental method
        exptl = entry_data.get('exptl', [])
        if exptl and len(exptl) > 0:
            data['experimental_method'] = exptl[0].get('method', 'Unknown')
        
        # Resolution - use correct field name: ls_dres_high
        # Also try rcsb_entry_info.resolution_combined as fallback
        refine = entry_data.get('refine', [])
        if refine and len(refine) > 0:
            resolution = refine[0].get('ls_dres_high')
            if resolution:
                data['resolution'] = f"{resolution} Å"
        
        # Fallback to resolution_combined if refine didn't have it
        if not data['resolution']:
            entry_info = entry_data.get('rcsb_entry_info', {})
            resolution_combined = entry_info.get('resolution_combined', [])
            if resolution_combined and len(resolution_combined) > 0:
                data['resolution'] = f"{resolution_combined[0]} Å"
        
        # Citation info from rcsb_primary_citation - use correct lowercase field names
        citations = entry_data.get('rcsb_primary_citation', {})
        if citations:
            data['pubmed_id'] = citations.get('pdbx_database_id_pub_med')
            data['doi'] = citations.get('pdbx_database_id_doi')
            data['authors'] = citations.get('rcsb_authors', [])
            
            # Build citation string
            journal = citations.get('journal_abbrev', '')
            year = citations.get('year')
            title = citations.get('title', '')
            data['citation'] = f"{title} {journal} ({year})" if title and year else None
        
        # Extract ligands from rcsb_entry_info.nonpolymer_bound_components
        entry_info = entry_data.get('rcsb_entry_info', {})
        ligand_components = entry_info.get('nonpolymer_bound_components', [])
        if ligand_components:
            data['ligands'] = ligand_components
        
        # Fetch PubMed data (abstract and mesh terms) from pubmed_core endpoint
        try:
            pub_url = f"https://data.rcsb.org/rest/v1/core/pubmed/{pdb_id}"
            pub_response = requests.get(pub_url, timeout=10)
            if pub_response.status_code == 200:
                try:
                    pub_data = pub_response.json()
                except Exception:
                    pub_data = {}
                
                # Get abstract from pubmed_core
                data['pubmed_abstract'] = pub_data.get('rcsb_pubmed_abstract_text')
                
                # Get MeSH terms
                data['mesh_terms'] = pub_data.get('rcsb_pubmed_mesh_descriptors', [])
                
                # Get pubmed_id from here if not found in entry
                if not data['pubmed_id']:
                    container_ids = pub_data.get('rcsb_pubmed_container_identifiers', {})
                    data['pubmed_id'] = container_ids.get('pubmed_id')
                
                # Get DOI from here if not found in entry
                if not data['doi']:
                    data['doi'] = pub_data.get('rcsb_pubmed_doi')
        except Exception as e:
            print(f"Error fetching pubmed_core data: {e}")
            
    except Exception as e:
        print(f"Error fetching RCSB data for {pdb_id}: {e}")
    
    return data


def create_system_prompt(rcsb_data: Dict[str, Any]) -> str:
    """
    Create system prompt for Azure OpenAI with RCSB PDB context.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        
    Returns:
        System prompt string
    """
    prompt = """You are an expert RNA structural biologist assistant. You have access to detailed information about an RNA structure from the RCSB Protein Data Bank.

Your role is to:
1. Provide clear, accurate information about this RNA structure
2. Answer questions about structural features, experimental methods, biological function, and ligand interactions
3. Use the provided PDB data as your primary source, supplemented with your general knowledge of RNA biology
4. Be conversational but scientifically accurate
5. When you don't have specific information in the provided data, acknowledge this and provide general context

Here is the information about the current RNA structure:
"""
    
    # Add PDB ID and title
    prompt += f"\n**PDB ID:** {rcsb_data['pdb_id']}"
    if rcsb_data.get('title'):
        prompt += f"\n**Title:** {rcsb_data['title']}"
    
    # Add authors
    authors = rcsb_data.get('authors', [])
    if authors:
        prompt += f"\n**Authors:** {', '.join(authors)}"
    
    # Add experimental details
    if rcsb_data.get('experimental_method'):
        prompt += f"\n**Experimental Method:** {rcsb_data['experimental_method']}"
    if rcsb_data.get('resolution'):
        prompt += f"\n**Resolution:** {rcsb_data['resolution']}"
    
    # Add citation
    if rcsb_data.get('citation'):
        prompt += f"\n**Citation:** {rcsb_data['citation']}"
    if rcsb_data.get('doi'):
        prompt += f"\n**DOI:** {rcsb_data['doi']}"
    
    # Add abstract if available
    if rcsb_data.get('pubmed_abstract'):
        prompt += f"\n\n**Research Abstract:**\n{rcsb_data['pubmed_abstract']}"
    
    # Add ligand information (now a simple list of IDs)
    ligands = rcsb_data.get('ligands', [])
    if ligands:
        prompt += f"\n\n**Bound Ligands:** {', '.join(ligands)}"
    
    # Add MeSH terms for additional context
    mesh_terms = rcsb_data.get('mesh_terms', [])
    if mesh_terms:
        prompt += f"\n\n**MeSH Terms:** {', '.join(mesh_terms)}"
    
    # Add description
    if rcsb_data.get('description'):
        prompt += f"\n\n**Description:** {rcsb_data['description']}"
    
    prompt += "\n\nRemember to be helpful, accurate, and cite the specific data when relevant."
    
    return prompt


def generate_introduction_prompt(rcsb_data: Dict[str, Any]) -> str:
    """
    Generate a prompt to create an introduction message about the RNA structure.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        
    Returns:
        User prompt requesting introduction
    """
    prompt = f"""Please provide a brief, friendly introduction to this RNA structure (PDB ID: {rcsb_data['pdb_id']}). 

Keep it concise (3-4 sentences) and highlight:
- What this structure represents
- The experimental method used
- Key findings or significance

Then, end with: "I can answer questions about:" followed by a bullet list of 3 specific topics:
- Experimental details (method: {rcsb_data.get('experimental_method', 'N/A')}, resolution: {rcsb_data.get('resolution', 'N/A')})
- Ligand binding ({', '.join(rcsb_data.get('ligands', [])[:3]) or 'N/A'})
- Biological function and mechanism"""
    
    return prompt


def call_azure_openai(
    messages: List[Dict[str, str]], 
    api_key: str, 
    endpoint: str,
    deployment: str = "gpt-4"
) -> Optional[str]:
    """
    Call Azure OpenAI Chat Completions API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
        deployment: Deployment name (default: "gpt-4")
        
    Returns:
        Assistant's response text or None if error
    """
    try:
        from openai import AzureOpenAI
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-12-01-preview",
            azure_endpoint=endpoint
        )
        
        # Make the API call
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,  # ~150-200 words
            top_p=1.0
        )
        
        # Extract the assistant's response
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        
        return None
        
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        raise e


def generate_introduction(rcsb_data: Dict[str, Any], api_key: str, endpoint: str, deployment: str = "gpt-4") -> str:
    """
    Generate introduction message using Azure OpenAI.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
        deployment: Deployment name (default: "gpt-4")
        
    Returns:
        Introduction message text
    """
    try:
        system_prompt = create_system_prompt(rcsb_data)
        user_prompt = generate_introduction_prompt(rcsb_data)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = call_azure_openai(messages, api_key, endpoint, deployment)
        
        if response:
            return response
        else:
            # Fallback message if API fails
            return generate_fallback_introduction(rcsb_data)
            
    except Exception as e:
        print(f"Error generating introduction: {e}")
        return generate_fallback_introduction(rcsb_data)


def generate_fallback_introduction(rcsb_data: Dict[str, Any]) -> str:
    """
    Generate a basic introduction without AI when API is unavailable.
    
    Args:
        rcsb_data: Dictionary containing RCSB PDB information
        
    Returns:
        Basic introduction text
    """
    intro = f"Welcome! This is the structure of {rcsb_data.get('title', 'Unknown')} (PDB ID: {rcsb_data['pdb_id']})."
    
    if rcsb_data.get('experimental_method'):
        intro += f" The structure was determined using {rcsb_data['experimental_method']}"
        if rcsb_data.get('resolution'):
            intro += f" at {rcsb_data['resolution']} resolution"
        intro += "."
    
    # Ligands is now a simple list of strings like ["K", "MG", "TPP"]
    ligands = rcsb_data.get('ligands', [])
    if ligands:
        # Show up to 3 ligands
        ligand_names = ligands[:3]
        intro += f" This structure includes bound ligands: {', '.join(ligand_names)}."
    
    intro += "\n\nI can answer questions about:\n"
    intro += f"- Experimental details ({rcsb_data.get('experimental_method', 'N/A')}, {rcsb_data.get('resolution', 'N/A')})\n"
    ligands_str = ', '.join(rcsb_data.get('ligands', [])[:3]) or 'N/A'
    intro += f"- Ligand binding ({ligands_str})\n"
    intro += "- Biological function and mechanism"
    
    return intro

