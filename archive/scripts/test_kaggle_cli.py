import traceback
from kaggle.api.kaggle_api_extended import KaggleApi

print('Testing Kaggle API submit response...')
api = KaggleApi()
api.authenticate()
try:
    res = api.competition_submit('submission.csv', 'diagnostic check', 'tabular-long-form-competition-ugcml-harbour-space')
    print('Submit success:', res)
except Exception as e:
    print('Error type:', type(e))
    print('Error message:', str(e))
    if hasattr(e, 'response') and e.response is not None:
        print('Response status:', e.response.status_code)
        print('Response text:', e.response.text)
    else:
        traceback.print_exc()


