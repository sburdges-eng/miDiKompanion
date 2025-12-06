// Quick test script to verify EmotionWheel integration
// Run with: node test_emotion_wheel.js

async function testEmotionWheel() {
  console.log('🧪 Testing EmotionWheel Integration\n');
  
  // Test 1: API Health Check
  console.log('1️⃣ Testing API Health...');
  try {
    const health = await fetch('http://localhost:8000/health');
    const healthData = await health.json();
    if (healthData.status === 'healthy') {
      console.log('   ✅ API is healthy\n');
    } else {
      console.log('   ❌ API health check failed\n');
      return;
    }
  } catch (error) {
    console.log('   ❌ API not reachable:', error.message, '\n');
    return;
  }

  // Test 2: Fetch Emotions
  console.log('2️⃣ Fetching emotions from API...');
  try {
    const response = await fetch('http://localhost:8000/emotions');
    const data = await response.json();
    
    if (data.success && data.emotions) {
      const baseEmotions = Object.keys(data.emotions).filter(k => k !== 'blends');
      console.log(`   ✅ Loaded ${baseEmotions.length} base emotions: ${baseEmotions.join(', ')}\n`);
      
      // Test 3: Transform Data
      console.log('3️⃣ Testing data transformation...');
      const transformed = transformEmotionData(data);
      
      if (transformed && transformed.emotions) {
        const transformedBases = Object.keys(transformed.emotions);
        console.log(`   ✅ Transformed ${transformedBases.length} base emotions\n`);
        
        // Test 4: Verify Structure
        console.log('4️⃣ Verifying EmotionWheel structure...');
        let allValid = true;
        
        for (const base of transformedBases) {
          const emotion = transformed.emotions[base];
          if (!emotion.intensities) {
            console.log(`   ❌ Missing intensities for ${base}`);
            allValid = false;
            continue;
          }
          
          const intensities = Object.keys(emotion.intensities);
          if (intensities.length === 0) {
            console.log(`   ❌ No intensities for ${base}`);
            allValid = false;
            continue;
          }
          
          // Check each intensity has sub-emotions
          for (const intensity of intensities) {
            const subs = emotion.intensities[intensity];
            if (!Array.isArray(subs) || subs.length === 0) {
              console.log(`   ❌ No sub-emotions for ${base} -> ${intensity}`);
              allValid = false;
            }
          }
        }
        
        if (allValid) {
          console.log(`   ✅ All ${transformedBases.length} emotions have valid structure\n`);
          
          // Test 5: Sample Selection Path
          console.log('5️⃣ Testing sample emotion selection path...');
          const sampleBase = transformedBases[0];
          const sampleIntensities = Object.keys(transformed.emotions[sampleBase].intensities);
          const sampleIntensity = sampleIntensities[0];
          const sampleSubs = transformed.emotions[sampleBase].intensities[sampleIntensity];
          const sampleSub = sampleSubs[0];
          
          console.log(`   ✅ Sample path: ${sampleBase} → ${sampleIntensity} → ${sampleSub}\n`);
          
          console.log('🎉 All tests passed! EmotionWheel is ready to use.\n');
          console.log('📝 Next steps:');
          console.log('   1. Open http://localhost:1420/ in your browser');
          console.log('   2. Click "Side B" to switch to Therapeutic Interface');
          console.log('   3. Click "Load Emotions" button');
          console.log('   4. Select: Base → Intensity → Sub-emotion');
        } else {
          console.log('   ❌ Structure validation failed\n');
        }
      } else {
        console.log('   ❌ Transformation failed\n');
      }
    } else {
      console.log('   ❌ Invalid API response\n');
    }
  } catch (error) {
    console.log('   ❌ Error:', error.message, '\n');
  }
}

// Transform function (matches App.tsx)
function transformEmotionData(apiResponse) {
  if (!apiResponse || !apiResponse.emotions) {
    return null;
  }

  const transformed = {
    emotions: {},
    total_nodes: apiResponse.total_nodes || 0
  };

  Object.keys(apiResponse.emotions).forEach((baseKey) => {
    if (baseKey === 'blends') return;
    
    const baseEmotion = apiResponse.emotions[baseKey];
    transformed.emotions[baseKey] = {
      intensities: {}
    };

    if (baseEmotion.sub_emotions) {
      Object.keys(baseEmotion.sub_emotions).forEach((subKey) => {
        const subEmotion = baseEmotion.sub_emotions[subKey];
        if (subEmotion.sub_sub_emotions) {
          Object.keys(subEmotion.sub_sub_emotions).forEach((subSubKey) => {
            const subSubEmotion = subEmotion.sub_sub_emotions[subSubKey];
            if (subSubEmotion.intensity_tiers) {
              Object.keys(subSubEmotion.intensity_tiers).forEach((intensityKey) => {
                if (!transformed.emotions[baseKey].intensities[intensityKey]) {
                  transformed.emotions[baseKey].intensities[intensityKey] = [];
                }
                if (!transformed.emotions[baseKey].intensities[intensityKey].includes(subSubKey)) {
                  transformed.emotions[baseKey].intensities[intensityKey].push(subSubKey);
                }
              });
            }
          });
        }
      });
    }
  });

  return transformed;
}

// Run tests
testEmotionWheel().catch(console.error);
