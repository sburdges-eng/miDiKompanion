#!/bin/bash
# Setup and Start Auto Emotion Sampler
# Downloads samples for 6 base emotions × 4 instruments

echo "======================================================================"
echo "AUTO EMOTION-INSTRUMENT SAMPLER SETUP"
echo "======================================================================"
echo ""
echo "This will download FREE samples organized by:"
echo "  • 6 Base Emotions: HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST"
echo "  • 4 Instruments: piano, guitar, drums, vocals"
echo "  • 25MB per emotion-instrument combination"
echo ""
echo "Samples will be saved to:"
echo "  ~/sburdges@gmail.com - Google Drive/My Drive/iDAW_Samples/Emotion_Instrument_Library/"
echo ""
echo "======================================================================"
echo ""

# Check if API key already exists
if [ -f "freesound_config.json" ]; then
    echo "✓ Freesound API key found"
    echo ""
    read -p "Start downloading? (y/n): " start_now

    if [ "$start_now" = "y" ] || [ "$start_now" = "Y" ]; then
        echo ""
        echo "Starting automatic download..."
        echo "This will download samples in order:"
        echo "  1. HAPPY (piano, guitar, drums, vocals)"
        echo "  2. SAD (piano, guitar, drums, vocals)"
        echo "  3. ANGRY (piano, guitar, drums, vocals)"
        echo "  4. FEAR (piano, guitar, drums, vocals)"
        echo "  5. SURPRISE (piano, guitar, drums, vocals)"
        echo "  6. DISGUST (piano, guitar, drums, vocals)"
        echo "  7. Then sub-emotions..."
        echo ""
        sleep 2
        ./auto_emotion_sampler.py start
    else
        echo "Cancelled. Run './auto_emotion_sampler.py start' when ready."
    fi
else
    echo "⚠ Freesound API key required (one-time setup)"
    echo ""
    echo "STEP 1: Get FREE API key"
    echo "  1. Visit: https://freesound.org/"
    echo "  2. Create free account (2 minutes)"
    echo "  3. Go to: https://freesound.org/apiv2/apply/"
    echo "  4. Create API application:"
    echo "     Name: 'iDAW Sample Fetcher'"
    echo "     Description: 'Personal sample library'"
    echo "  5. Copy your API key"
    echo ""
    echo "STEP 2: Enter API key below"
    echo ""
    read -p "Paste your Freesound API key: " api_key

    if [ -n "$api_key" ]; then
        # Save to config
        echo "{\"freesound_api_key\": \"$api_key\"}" > freesound_config.json
        echo ""
        echo "✓ API key saved to freesound_config.json"
        echo ""

        read -p "Start downloading now? (y/n): " start_now

        if [ "$start_now" = "y" ] || [ "$start_now" = "Y" ]; then
            echo ""
            echo "Starting automatic download..."
            sleep 2
            ./auto_emotion_sampler.py start
        else
            echo ""
            echo "Setup complete! Run './auto_emotion_sampler.py start' when ready."
        fi
    else
        echo ""
        echo "✗ No API key provided. Run './setup_and_start.sh' again when ready."
    fi
fi
