from django.db import models


class CodeDocument(models.Model):
    url = models.URLField(max_length=2048)
    repo = models.CharField(max_length=256)
    file_path = models.CharField(max_length=2048)
    identifier = models.CharField(max_length=2048)
    language = models.CharField(max_length=32)
    code = models.TextField()
    code_hash = models.CharField(max_length=64)
    # Used to index into the per-language code embedding matrix
    embedded_row_index = models.IntegerField()

    class Meta:
        indexes = [
            models.Index(fields=['language', 'embedded_row_index']),
            models.Index(fields=['code_hash']),
        ]

    def __str__(self):
        identifier = self.identifier if len(self.identifier) > 0 else self.code[:32]
        return f'({self.language}) {identifier}'


class QueryLog(models.Model):
    query = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.query} {self.created_at}'


class CodeDocumentVisitLog(models.Model):
    code_document = models.ForeignKey(CodeDocument, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.code_document.file_path}:{self.code_document.identifier} {self.created_at}'


class CodeDocumentQueryRating(models.Model):
    code_document = models.ForeignKey(CodeDocument, on_delete=models.CASCADE)
    query = models.TextField()
    rating = models.IntegerField()
    rater_hash = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.query}:{self.code_document.identifier}:{self.rating} {self.created_at}'
