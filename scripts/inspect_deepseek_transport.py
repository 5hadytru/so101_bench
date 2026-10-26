"""Metered stock Inspect Responses transport restricted to DeepSeek Flash-1017."""
import json
import httpx
from inspect_budget_transport import BudgetTransport,BudgetExceeded


class DeepSeekBudgetTransport(BudgetTransport):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # DeepSeek counts hidden reasoning against this allowance. At 4096
        # tokens native camera/calibration observations repeatedly exhausted
        # it before a tool call. Dollar and wall-time guards still apply.
        self.output_cap=32768

    def handle_request(self,request):
        body=json.loads(request.content)
        assert request.url.host=='api.deepseek.com' and body['model']=='deepseek-flash'
        body['max_output_tokens']=self.output_cap
        body['store']=False
        # Conservative image reserve; provider-specific reasoning allowance.
        def bound(value):
            if isinstance(value,dict):
                if value.get('type')=='input_image':return 6144
                return sum(bound(v) for v in value.values())
            if isinstance(value,list):return sum(bound(v) for v in value)
            return len(value.encode()) if isinstance(value,str) else 0
        reserved_tokens=bound(body)
        reserve=(reserved_tokens*.15+self.output_cap*.6)/1e6
        if self.spent()+reserve>self.total_limit or self.spent(self.trial)+reserve>self.trial_limit:
            raise BudgetExceeded('DeepSeek Direct budget reserve reached; no request sent')
        effective=httpx.Request(request.method,request.url,headers=request.headers,content=json.dumps(body).encode(),extensions=request.extensions)
        effective.headers['Content-Length']=str(len(effective.content))
        response=self.inner.handle_request(effective);response.read()
        self.ledger.parent.mkdir(parents=True,exist_ok=True)
        if response.status_code>=400:
            with self.ledger.with_suffix('.errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(trial=self.trial,http_status=response.status_code,
                    error_code='credit_balance_exhausted' if response.status_code==402 else None))+'\n')
        elif response.status_code==200:
            data=response.json();usage=data.get('usage') or {}
            inp=int(usage.get('input_tokens') or 0);out=int(usage.get('output_tokens') or 0)
            cached=int((usage.get('input_tokens_details') or {}).get('cached_tokens') or 0)
            cost=(max(0,inp-cached)*.15+cached*.003+out*.6)/1e6 if usage else reserve
            with self.ledger.open('a') as f:f.write(json.dumps(dict(trial=self.trial,response_id=data.get('id'),usage=usage,estimated_usd=cost,
                reserved_input_tokens=reserved_tokens,reserved_usd=reserve,max_output_tokens=self.output_cap,
                provider='deepseek-flash-1017',pricing_rates=[.15,.003,.6]))+'\n')
            if not usage:raise BudgetExceeded('DeepSeek omitted usage; reserve recorded, stopping')
        return response
