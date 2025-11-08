import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import os

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
    
    def chat(self, message: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
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
            return "❌ Model is not loaded."
        
        try:
            # Format prompt in Gemma chat format
            prompt = f"""<start_of_turn>user
{message}<end_of_turn>
<start_of_turn>model
"""
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to the same device as model
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
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
            
            return bot_response if bot_response else "I'm sorry, I couldn't generate a response."
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history (for future implementation)."""
        pass  # Can be extended for multi-turn conversations


def main():
    """Main function to run the chatbot."""
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

