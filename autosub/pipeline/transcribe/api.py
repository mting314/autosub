import logging
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

logger = logging.getLogger(__name__)


def transcribe_uri(
    gcs_uri: str,
    project_id: str,
    language_codes: list[str] | None = None,
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    replacements: dict[str, str] | None = None,
) -> speech_v2.BatchRecognizeResponse:
    """
    Sends a long-running batch transcription request to Chirp 3 using a GCS URI.
    Required for audio files longer than 1 minute.
    """
    if not language_codes:
        language_codes = ["ja-JP"]
    client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint="us-speech.googleapis.com")
    )

    features = speech_v2.RecognitionFeatures(
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )
    if num_speakers is not None and num_speakers > 0:
        features.diarization_config = cloud_speech.SpeakerDiarizationConfig(
            min_speaker_count=num_speakers,
            max_speaker_count=num_speakers,
        )
        logger.info(f"Speaker diarization enabled with {num_speakers} speaker(s)")

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=language_codes,
        model="chirp_3",
        features=features,
    )

    if replacements:
        config.transcript_normalization = cloud_speech.TranscriptNormalization(
            entries=[
                cloud_speech.TranscriptNormalization.Entry(
                    search=search, replace=replace, case_sensitive=False,
                )
                for search, replace in replacements.items()
            ]
        )
        logger.info(f"Transcript normalization enabled with {len(replacements)} entries")

    # Note: SpeechAdaptation (phrase_sets) is not supported on Chirp 3.
    # Vocabulary hints are accepted by the CLI but not passed to the API.
    if vocabulary:
        logger.debug(
            "Vocabulary hints ignored — Chirp 3 does not support SpeechAdaptation"
        )

    request = speech_v2.BatchRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/us/recognizers/_",
        config=config,
        files=[speech_v2.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=speech_v2.RecognitionOutputConfig(
            inline_response_config=speech_v2.InlineOutputConfig()
        ),
    )

    logger.info(f"Starting long-running transcription on {gcs_uri}...")
    operation = client.batch_recognize(request=request)

    # Wait for the operation to complete
    response = operation.result()
    logger.info("Transcription complete!")
    return response  # type: ignore


def transcribe_local_file(
    audio_content: bytes,
    project_id: str,
    language_codes: list[str] | None = None,
    vocabulary: list[str] | None = None,
    num_speakers: int | None = None,
    replacements: dict[str, str] | None = None,
) -> speech_v2.RecognizeResponse:
    """
    Sends a standard synchronous transcription request using local audio bytes.
    Can only be used if audio is strictly < 1 minute.
    """
    if not language_codes:
        language_codes = ["ja-JP"]
    client = speech_v2.SpeechClient(
        client_options=ClientOptions(api_endpoint="us-speech.googleapis.com")
    )

    features = speech_v2.RecognitionFeatures(
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )
    if num_speakers is not None and num_speakers > 0:
        features.diarization_config = cloud_speech.SpeakerDiarizationConfig(
            min_speaker_count=num_speakers,
            max_speaker_count=num_speakers,
        )
        logger.info(f"Speaker diarization enabled with {num_speakers} speaker(s)")

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=language_codes,
        model="chirp_3",
        features=features,
    )

    if replacements:
        config.transcript_normalization = cloud_speech.TranscriptNormalization(
            entries=[
                cloud_speech.TranscriptNormalization.Entry(
                    search=search, replace=replace, case_sensitive=False,
                )
                for search, replace in replacements.items()
            ]
        )
        logger.info(f"Transcript normalization enabled with {len(replacements)} entries")

    # Note: SpeechAdaptation (phrase_sets) is not supported on Chirp 3.
    # Vocabulary hints are accepted by the CLI but not passed to the API.
    if vocabulary:
        logger.debug(
            "Vocabulary hints ignored — Chirp 3 does not support SpeechAdaptation"
        )

    request = speech_v2.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/us/recognizers/_",
        config=config,
        content=audio_content,
    )

    logger.info("Starting fast synchronous transcription...")
    response = client.recognize(request=request)
    logger.info("Transcription complete!")
    return response
