from django.views.generic import TemplateView

from code_search_app import models
from code_search_app.shared import get_pygments_html_formatter


def get_syntax_highlight_css():
    return get_pygments_html_formatter().get_style_defs('.codesnippetsearch-highlight')


class IndexView(TemplateView):
    template_name = 'dist/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['syntax_highlight_css'] = get_syntax_highlight_css()
        context['n_repositories'] = models.CodeRepository.objects.count()
        return context
