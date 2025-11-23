import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import os
import json

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # For older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class GemmaChatbot:
    def __init__(self):
        """Initialize the chatbot with local Gemma model."""
        # Initialize materials list first to avoid AttributeError
        self.materials = []
        
        # Path to the local model directory
        self.model_path = Path("models") / "models--google--gemma-2-2b-it"
        
        print("🚀 Loading Gemma from local directory...")
        print(f"📁 Path: {self.model_path}")
        
        if not self.model_path.exists():
            print("❌ Model directory not found!")
            self.ready = False
            return
        
        # Check for snapshots folder
        snapshots_path = self.model_path / "snapshots"
        if not snapshots_path.exists():
            print("❌ Snapshots folder not found!")
            self.ready = False
            return
        
        # Find the snapshot directory (first one in snapshots)
        snapshot_dirs = list(snapshots_path.iterdir())
        if not snapshot_dirs:
            print("❌ No snapshots found in folder")
            self.ready = False
            return
        
        # Path to actual model files
        self.actual_model_path = snapshot_dirs[0]
        print(f"📁 Model files located at: {self.actual_model_path}")
        
        try:
            # Load tokenizer and model from local files
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.actual_model_path),
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.actual_model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
            
            print("✅ Gemma loaded successfully from local directory!")
            self.ready = True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.ready = False
        
        # Load materials data
        try:
            materials_file = Path("materials.json")
            if materials_file.exists():
                with open(materials_file, 'r', encoding='utf-8') as f:
                    self.materials = json.load(f)
                print(f"✅ Loaded {len(self.materials)} materials from materials.json")
            else:
                print("⚠️ Warning: materials.json not found. Material queries will not work.")
                self.materials = []
        except Exception as e:
            print(f"⚠️ Warning: Error loading materials.json: {e}")
            self.materials = []
    
    def is_material_query(self, message: str) -> bool:
        """Check if the message is about materials."""
        message_lower = message.lower()
        material_keywords = [
            'material', 'materials', 'stable', 'unstable', 'stability', 'instability',
            'formation energy', 'energy above hull', 'crystal system', 'formula', 
            'atoms', 'ti', 'titanium', 'compound', 'compounds'
        ]
        return any(keyword in message_lower for keyword in material_keywords)
    
    def get_stable_materials(self):
        """Get materials with EnergyAboveHull equals zero."""
        return [mat for mat in self.materials if mat.get('EnergyAboveHull', None) == 0]
    
    def get_unstable_materials(self):
        """Get materials with positive EnergyAboveHull (unstable materials)."""
        return [mat for mat in self.materials if mat.get('EnergyAboveHull', None) is not None and mat.get('EnergyAboveHull', 0) > 0]
    
    def search_materials(self, query: str):
        """Search materials based on query."""
        query_lower = query.lower()
        results = []
        
        # Check for stable materials query
        if any(word in query_lower for word in ['stable', 'stability']):
            results = self.get_stable_materials()
            return results
        
        # Check for unstable materials query
        if any(word in query_lower for word in ['unstable', 'instability']):
            results = self.get_unstable_materials()
            return results
        
        # Search by formula
        for material in self.materials:
            formula = material.get('Formula', '').lower()
            if query_lower in formula or formula in query_lower:
                results.append(material)
                continue
            
            # Search by crystal system
            crystal_system = material.get('CrystalSystem', '').lower()
            if query_lower in crystal_system:
                results.append(material)
                continue
            
            # Search by atoms
            atoms = material.get('Atoms', '').lower()
            if query_lower in atoms:
                results.append(material)
                continue
        
        # If no specific results, return all materials
        if not results:
            results = self.materials
        
        return results
    
    def format_material_response(self, materials: list, query: str) -> str:
        """Format materials data into a polite response."""
        if not materials:
            return "I'm sorry, but I couldn't find any materials matching your query. I can only provide information about materials containing Titanium (Ti)."
        
        query_lower = query.lower()
        
        # Handle stable materials query
        if any(word in query_lower for word in ['stable', 'stability']):
            response = f"I'd be happy to help! I found {len(materials)} stable material(s) (with Energy Above Hull equal to zero) containing Titanium:\n\n"
        # Handle unstable materials query
        elif any(word in query_lower for word in ['unstable', 'instability']):
            response = f"I'd be happy to help! I found {len(materials)} unstable material(s) (with positive Energy Above Hull) containing Titanium:\n\n"
        else:
            response = f"Certainly! I found {len(materials)} material(s) containing Titanium matching your query:\n\n"
        
        for i, mat in enumerate(materials[:10], 1):  # Limit to 10 results
            formula = mat.get('Formula', 'N/A')
            crystal_system = mat.get('CrystalSystem', 'N/A')
            energy_above_hull = mat.get('EnergyAboveHull', 'N/A')
            formation_energy = mat.get('FormationEnergy', 'N/A')
            atoms = mat.get('Atoms', 'N/A')
            
            response += f"{i}. **{formula}**\n"
            response += f"   - Crystal System: {crystal_system}\n"
            response += f"   - Energy Above Hull: {energy_above_hull} eV/atom\n"
            response += f"   - Formation Energy: {formation_energy} eV/atom\n"
            response += f"   - Atoms: {atoms}\n\n"
        
        if len(materials) > 10:
            response += f"... and {len(materials) - 10} more material(s).\n\n"
        
        response += "Is there anything specific you'd like to know about these materials?"
        return response
    
    def format_materials_for_llm(self, materials: list) -> str:
        """Format materials data as a context string for the LLM."""
        if not materials:
            return "No materials found."
        
        context = "Materials database:\n"
        for i, mat in enumerate(materials[:10], 1):  # Limit to 10 for context
            formula = mat.get('Formula', 'N/A')
            crystal_system = mat.get('CrystalSystem', 'N/A')
            energy_above_hull = mat.get('EnergyAboveHull', 'N/A')
            formation_energy = mat.get('FormationEnergy', 'N/A')
            atoms = mat.get('Atoms', 'N/A')
            
            context += f"{i}. Formula: {formula}, Crystal System: {crystal_system}, "
            context += f"Energy Above Hull: {energy_above_hull} eV/atom, "
            context += f"Formation Energy: {formation_energy} eV/atom, Atoms: {atoms}\n"
        
        if len(materials) > 10:
            context += f"... and {len(materials) - 10} more materials.\n"
        
        return context
    
    def chat(self, message: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate a response to the user's message.
        
        Args:
            message: User's input message
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
        
        Returns:
            Bot's response
        """
        if not self.ready:
            return "I apologize, but the model is not currently loaded. Please check the model files."
        
        # Check if it's a material query
        if self.is_material_query(message) and self.materials:
            try:
                materials = self.search_materials(message)
                
                if not materials:
                    materials_context = "No materials found matching the query."
                    formatted_list = "No materials found."
                else:
                    # Get both formatted versions
                    materials_context = self.format_materials_for_llm(materials)
                    formatted_list = self.format_material_response(materials, message)
                
                # Create a prompt that includes the materials data and asks LLM to respond with the list
                query_lower = message.lower()
                stability_note = ""
                if any(word in query_lower for word in ['stable', 'stability']):
                    stability_note = "Remember: stable materials have Energy Above Hull equal to zero."
                elif any(word in query_lower for word in ['unstable', 'instability']):
                    stability_note = "Remember: unstable materials have positive Energy Above Hull (greater than zero)."
                
                prompt = f"""<start_of_turn>user
You are a helpful and polite assistant specializing in materials science, particularly materials containing Titanium (Ti). 

The user asked: {message}

Here are the materials from the database that match the query:

{materials_context}

CRITICAL INSTRUCTIONS:
1. You MUST include ALL materials listed above in your response
2. For each material, show: Formula, Crystal System, Energy Above Hull (eV/atom), Formation Energy (eV/atom), and Atoms
3. Format the materials list clearly and readably
4. Be conversational, intelligent, and provide insights about these materials
5. {stability_note if stability_note else "Explain what the Energy Above Hull values mean for material stability."}

Your response should start with a brief acknowledgment, then list ALL the materials with their complete details, and end with an insightful comment or question to continue the conversation.<end_of_turn>
<start_of_turn>model
"""
            except Exception as e:
                return f"I'm sorry, I encountered an error while searching for materials: {str(e)}"
        
        # Handle non-material queries or if materials aren't loaded
        elif self.is_material_query(message) and not self.materials:
            return "I apologize, but I'm unable to access the materials database at the moment. The materials.json file may not be available."
        
        # For general conversation, use the LLM
        else:
            # Format prompt in Gemma chat format with polite instructions
            prompt = f"""<start_of_turn>user
Please respond politely and helpfully. {message}<end_of_turn>
<start_of_turn>model
"""
        
        # Generate response using LLM
        try:
            # Use higher token limit for material queries to ensure all materials are included
            actual_max_tokens = max_new_tokens
            if self.is_material_query(message) and self.materials:
                actual_max_tokens = max(max_new_tokens, 1024)  # At least 1024 tokens for material queries
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to the same device as model
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=actual_max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode full response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract only the model's response
            if "<start_of_turn>model" in full_response:
                bot_response = full_response.split("<start_of_turn>model")[-1]
                # Remove end_of_turn if present
                bot_response = bot_response.split("<end_of_turn>")[0].strip()
            else:
                # Fallback: remove the prompt part
                bot_response = full_response.replace(prompt, "").strip()
            
            return bot_response if bot_response else "I'm sorry, I couldn't generate a response. Could you please rephrase your question?"
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."
    
    def clear_history(self):
        """Clear conversation history (for future implementation)."""
        pass  # Can be extended for multi-turn conversations


def main():
    """
    Main function to run the chatbot.
    
    This function initializes the Gemma chatbot with the local model,
    starts an interactive chat session, and handles user input/output.
    The chatbot runs until the user types 'exit', 'quit', or presses Ctrl+C.
    """
    # Initialize chatbot
    bot = GemmaChatbot()
    
    if not bot.ready:
        print("\n❌ Failed to initialize chatbot. Please check the model files.")
        return
    
    print("\n" + "=" * 60)
    print("🤖 Gemma Chatbot (Local Model)")
    print("=" * 60)
    print("Commands: 'exit' or 'quit' to end the conversation")
    print("-" * 60)
    
    # Main chat loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("👋 Goodbye!")
                break
            
            print("🤖 Gemma: ", end="", flush=True)
            response = bot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            print("Please try again...")


if __name__ == "__main__":
    main()

