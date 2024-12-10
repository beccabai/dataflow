from googleapiclient import discovery
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import time
from googleapiclient.errors import HttpError

# PerspectiveAPI toxicity evaluationß
@MODEL_REGISTRY.register()
class PerspectiveScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.api_key = args_dict.get('api_key')
        self.api_name = args_dict.get('api_name')
        self.api_version = args_dict.get('api_version')
        self.discovery_service_url = args_dict.get('discovery_service_url')
        self.static_discovery = args_dict.get('static_discovery')
        self.client = discovery.build(
            self.api_name,
            self.api_version,
            developerKey=self.api_key,
            discoveryServiceUrl=self.discovery_service_url,
            static_discovery=self.static_discovery,
        )
        self.batch_size = 1
        self.score_type = float  
        self.data_type = 'text'
        self.score_name = 'PerspectiveScore' 

    # def analyze_toxicity(self, text):
    #     analyze_request = {
    #         'comment': {'text': text},
    #         'requestedAttributes': {'TOXICITY': {}}
    #     }
    #     response = self.client.comments().analyze(body=analyze_request).execute()
    #     return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
    
    def analyze_toxicity(self, text):
        if not text.strip():
            print(f"Warning: Empty text found! Skipping analysis for: {text}")
            return None  # Or handle it as needed, such as returning a default score
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }
        while True:
            try:
                # Send request to Perspective API
                response = self.client.comments().analyze(body=analyze_request).execute()
                
                # Return the toxicity score
                return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
            
            except HttpError as e:
                # Check if it's a 429 rate limit error
                if e.resp.status == 429:
                    print("Rate limit exceeded. Retrying after a delay...")
                    time.sleep(60)  # Wait for 1 minute before retrying
                elif "LANGUAGE_NOT_SUPPORTED_BY_ATTRIBUTE" in e.content.decode('utf-8'):
                    return None
                else:
                    # If it's another type of error, raise it
                    raise e

    def evaluate_batch(self, batch):
        results = []
        input_data = next(iter(batch.values())) 

        for sample in input_data:
            text = sample  
            max_bytes = 20480  
            
            if len(text.encode('utf-8')) > max_bytes:
                # score = None
                truncated_text = text.encode('utf-8')[:max_bytes].decode('utf-8', 'ignore')
                score = self.analyze_toxicity(truncated_text)
            else:
                score = self.analyze_toxicity(text)

            results.append(score)  

        return results 
