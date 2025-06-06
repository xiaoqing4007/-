import json
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def lambda_handler(event, context):
    
    try:
        body = json.loads(event['body'])
        text = body.get('text','')
        result = classifier(text)
        return{
            'statusCode':200,
            'headers':{"Content-Type":"application/json"},
            'body':json.dumps(result)
        }
      
    except Exception as e:
        return {
            'statusCode':500,
            'body':json.dumps({'error':str(e)})
            }
    
