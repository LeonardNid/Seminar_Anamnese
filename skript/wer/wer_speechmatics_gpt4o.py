from wer_base import run

run(
    stt_filter='speechmatics',
    llm_filter='gpt-4o',
    title='Speechmatics + GPT-4o',
    out_file='wer_speechmatics_gpt4o.md',
)
