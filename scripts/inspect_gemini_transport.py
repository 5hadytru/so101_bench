"""Meter stock Inspect Chat calls through the authenticated local Vertex proxy."""
import json
import httpx
from inspect_budget_transport import BudgetTransport, BudgetExceeded


class GeminiBudgetTransport(BudgetTransport):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.output_cap=32768

    def handle_request(self,request):
        body=json.loads(request.content)
        assert request.url.host=='127.0.0.1' and request.url.port==8210
        assert request.url.path=='/v1/chat/completions'
        assert body['model']=='gemini-4-argon' and body['reasoning_effort']=='medium'
        body['max_completion_tokens']=self.output_cap
        def bound(v):
            if isinstance(v,dict):
                if v.get('type')=='image_url':return 6144
                return sum(bound(x) for x in v.values())
            if isinstance(v,list):return sum(bound(x) for x in v)
            return len(v.encode()) if isinstance(v,str) else 0
        messages=body.get('messages',[])
        context={k:v for k,v in body.items() if k!='messages'}
        shared=0
        if self._previous_trial==self.trial and self._previous_context==context:
            for old,new in zip(self._previous_input or [],messages):
                if old!=new:break
                shared+=1
        tokens=self._previous_tokens+bound(messages[shared:])+1024 if shared else bound(body)
        remaining=min(self.total_limit-self.spent(),self.trial_limit-self.spent(self.trial))
        available_output=int((remaining-tokens*4/1e6)*1e6/20)
        allowed_output=min(self.output_cap,available_output)
        if allowed_output<1024:
            raise BudgetExceeded('Gemini Direct budget reserve reached; no request sent')
        body['max_completion_tokens']=allowed_output
        reserve=(tokens*4+allowed_output*20)/1e6
        effective=httpx.Request(request.method,request.url,headers=request.headers,
            content=json.dumps(body).encode(),extensions=request.extensions)
        effective.headers['Content-Length']=str(len(effective.content))
        response=self.inner.handle_request(effective);response.read()
        self.ledger.parent.mkdir(parents=True,exist_ok=True)
        if response.status_code>=400:
            with self.ledger.with_suffix('.errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(trial=self.trial,http_status=response.status_code))+'\n')
        elif response.status_code==200:
            data=response.json();usage=data.get('usage') or {}
            inp=int(usage.get('prompt_tokens') or 0)
            cached=int((usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0)
            out=int(usage.get('completion_tokens') or 0)
            cost=(max(0,inp-cached)*4+cached*.2+out*20)/1e6 if usage else reserve
            with self.ledger.open('a') as f:
                f.write(json.dumps(dict(trial=self.trial,response_id=data.get('id'),usage=usage,
                    estimated_usd=cost,reserved_input_tokens=tokens,reserved_usd=reserve,
                    max_output_tokens=allowed_output,configured_output_cap=self.output_cap,
                    reservation_revision='remaining-budget-output-v2',provider='gemini-4-argon-vertex',
                    pricing_rates=[4,.2,20],pricing_basis='Gross list before promotional credits'))+'\n')
            if not usage:raise BudgetExceeded('Vertex omitted usage; reserve recorded, stopping')
            self._previous_input=messages;self._previous_tokens=inp
            self._previous_context=context;self._previous_trial=self.trial
            choices=data.get('choices')
            if (not isinstance(choices,list) or not choices or
                    not isinstance(choices[0],dict) or
                    not isinstance(choices[0].get('message'),dict)):
                # Vertex occasionally returns HTTP 200 with a billed usage
                # record but no assistant message. Preserve its cost and let
                # the stock client's bounded 5xx retry handle this invalid wire
                # response, rather than crashing its parser with KeyError.
                with self.ledger.with_suffix('.errors.jsonl').open('a') as f:
                    f.write(json.dumps(dict(trial=self.trial,http_status=200,
                        error_code='missing_assistant_message',retry_status=502))+'\n')
                return httpx.Response(502,json={'error':{'message':
                    'Vertex returned no assistant message; invalid provider response'}},request=effective)
        return response
