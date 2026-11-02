import ipaddress
from django.http import HttpResponseForbidden


class LoopbackOnly:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            if not ipaddress.ip_address(request.META.get("REMOTE_ADDR", "")).is_loopback:
                return HttpResponseForbidden("This app is available only on this computer.")
        except ValueError:
            return HttpResponseForbidden("This app is available only on this computer.")
        response = self.get_response(request)
        response["Cache-Control"] = "no-store"
        return response
