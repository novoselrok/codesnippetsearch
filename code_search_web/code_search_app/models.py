from django.db import models


class CodeLanguage(models.Model):
    name = models.CharField(max_length=32)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class CodeRepository(models.Model):
    UPDATE_IN_PROGRESS = 0
    UPDATE_FINISHED = 1
    UPDATE_ERROR = 2

    organization = models.CharField(max_length=256)
    name = models.CharField(max_length=256)
    description = models.TextField(blank=True, null=True)
    commit_hash = models.CharField(max_length=256, null=True, blank=True)
    update_status = models.IntegerField(choices=[
        (UPDATE_IN_PROGRESS, 'In Progress'),
        (UPDATE_FINISHED, 'Finished'),
        (UPDATE_ERROR, 'Error'),
    ], default=UPDATE_FINISHED)

    languages = models.ManyToManyField(CodeLanguage)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.organization}/{self.name}'


class CodeDocument(models.Model):
    url = models.URLField(max_length=2048)
    path = models.CharField(max_length=2048)
    identifier = models.CharField(max_length=2048)
    code = models.TextField()
    code_hash = models.CharField(max_length=64)
    # Used to index into code embedding matrix
    embedded_row_index = models.IntegerField()

    repository = models.ForeignKey(CodeRepository, on_delete=models.CASCADE)
    language = models.ForeignKey(CodeLanguage, on_delete=models.CASCADE)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['repository', 'language', 'embedded_row_index']),
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
