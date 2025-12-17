"""
Unified Framework

Integrates CIF, LAS, Resonant Ethics, and QEF into a single API.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
from datetime import datetime


@dataclass
class FrameworkConfig:
    """Configuration for the unified framework."""
    enable_cif: bool = True
    enable_las: bool = True
    enable_ethics: bool = True
    enable_qef: bool = True
    qef_node_id: Optional[str] = None
    ethics_strict_mode: bool = True


class UnifiedFramework:
    """
    Unified Framework
    
    Integrates all components (CIF, LAS, Resonant Ethics, QEF)
    into a single cohesive system.
    """
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize Unified Framework.
        
        Args:
            config: Framework configuration
        """
        self.config = config or FrameworkConfig()
        
        # Initialize components based on config
        if self.config.enable_cif:
            from ..cif import CIF
            self.cif = CIF()
        else:
            self.cif = None
        
        if self.config.enable_las:
            from ..las import LAS
            self.las = LAS()
        else:
            self.las = None
        
        if self.config.enable_ethics:
            from ..ethics import ResonantEthics, EmotionalConsentProtocol
            self.ethics = ResonantEthics()
            self.ecp = EmotionalConsentProtocol()
        else:
            self.ethics = None
            self.ecp = None
        
        if self.config.enable_qef:
            from ..qef import QEF
            self.qef = QEF(node_id=self.config.qef_node_id)
        else:
            self.qef = None
        
        # Integration state
        self.session_history: List[Dict] = []
        
    def create_with_consent(
        self,
        human_emotional_input: Dict,
        creative_goal: Optional[Dict] = None,
        require_consent: bool = True
    ) -> Dict:
        """
        Create art with ethical consent protocol.
        
        Args:
            human_emotional_input: Human emotional data
            creative_goal: Optional creative goal
            require_consent: Whether to require consent
        
        Returns:
            Creation result
        """
        # Step 1: Ethical consent (if enabled)
        if self.ecp and require_consent:
            # System declares state
            if self.las:
                # Get LAS emotional state
                las_esv = self.las.ei.process_emotional_input({})
                las_esv_dict = las_esv.to_dict()
            else:
                las_esv_dict = {"valence": 0.0, "arousal": 0.5, "dominance": 0.5, "tension": 0.5}
            
            self.ecp.system_declare_state(
                esv=las_esv_dict,
                intensity=0.5,
                stability=0.7
            )
            
            # Human declares intent
            human_intent = human_emotional_input.get("intent", {})
            self.ecp.human_declare_intent(
                emotional_intent=human_emotional_input,
                intent_type=human_intent.get("type", "creation")
            )
            
            # Evaluate consent
            consent_result = self.ecp.evaluate_consent()
            
            if not consent_result.get("consent_granted", False):
                return {
                    "error": "Consent denied",
                    "consent_result": consent_result,
                    "created": False
                }
        
        # Step 2: CIF integration (if enabled)
        if self.cif and self.las:
            # Get LAS emotional state
            las_esv = self.las.ei.process_emotional_input({})
            las_emotional_state = {"esv": las_esv.to_dict()}
            
            # Integrate via CIF
            cif_result = self.cif.integrate(
                human_bio_data=human_emotional_input,
                las_emotional_state=las_emotional_state
            )
        else:
            cif_result = None
        
        # Step 3: LAS generation
        if self.las:
            las_result = self.las.generate(
                emotional_input=human_emotional_input,
                creative_goal=creative_goal
            )
        else:
            las_result = {"error": "LAS not enabled"}
        
        # Step 4: QEF emission (if enabled)
        if self.qef and las_result.get("esv"):
            qas = self.qef.emit_emotional_state(
                esv=las_result["esv"],
                source="unified_framework"
            )
            qef_result = {"qas": qas.to_dict()}
        else:
            qef_result = None
        
        # Step 5: Ethical evaluation
        if self.ethics:
            # Evaluate ethical pillars
            ethics_scores = {
                "pillar_1": self.ethics.evaluate_pillar_1_sympathetic_autonomy(
                    system_autonomy_level=0.7 if self.las else 0.0,
                    forced_output=False
                ),
                "pillar_2": self.ethics.evaluate_pillar_2_emotional_transparency(
                    disclosure_level=0.8,
                    manipulation_intent=False
                ),
                "pillar_3": self.ethics.evaluate_pillar_3_mutual_evolution(
                    human_benefit=0.7,
                    ai_benefit=0.6
                ),
                "pillar_4": self.ethics.evaluate_pillar_4_harmonic_accountability(
                    resonance_effect=0.5,
                    reversibility=0.8
                ),
                "pillar_5": self.ethics.evaluate_pillar_5_aesthetic_stewardship(
                    creation_responsibility=0.7,
                    nurturing_level=0.6
                )
            }
            
            overall_ethics = self.ethics.get_overall_ethics_score()
        else:
            ethics_scores = None
            overall_ethics = None
        
        # Compile result
        result = {
            "created": True,
            "timestamp": datetime.now().isoformat(),
            "las_output": las_result,
            "cif_integration": cif_result,
            "qef_emission": qef_result,
            "ethics_scores": ethics_scores,
            "overall_ethics": overall_ethics,
            "consent_granted": consent_result.get("consent_granted", True) if self.ecp else True
        }
        
        # Store in history
        self.session_history.append(result)
        
        return result
    
    def evolve_from_feedback(self, feedback: Dict) -> Dict:
        """
        Evolve system from feedback.
        
        Args:
            feedback: Feedback data
        
        Returns:
            Evolution result
        """
        results = {}
        
        # LAS evolution
        if self.las:
            las_evolution = self.las.evolve(feedback)
            results["las_evolution"] = las_evolution
        
        # Ethics evaluation of feedback
        if self.ethics:
            # Check if feedback indicates harm
            harm_level = feedback.get("harm", 0.0)
            manipulation = feedback.get("manipulation", 0.0)
            
            if harm_level > 0.3 or manipulation > 0.3:
                # Ethical violation detected
                results["ethics_warning"] = {
                    "harm_level": harm_level,
                    "manipulation": manipulation,
                    "action": "review_required"
                }
        
        return results
    
    def get_collective_resonance(self) -> Dict:
        """
        Get collective resonance from QEF.
        
        Returns:
            Collective resonance data
        """
        if not self.qef:
            return {"error": "QEF not enabled"}
        
        return self.qef.receive_collective_resonance()
    
    def get_status(self) -> Dict:
        """Get unified framework status."""
        status = {
            "config": {
                "cif_enabled": self.config.enable_cif,
                "las_enabled": self.config.enable_las,
                "ethics_enabled": self.config.enable_ethics,
                "qef_enabled": self.config.enable_qef
            },
            "session_count": len(self.session_history)
        }
        
        if self.cif:
            status["cif_status"] = self.cif.get_status()
        
        if self.las:
            status["las_status"] = self.las.get_status()
        
        if self.qef:
            status["qef_status"] = self.qef.get_status()
        
        if self.ethics:
            status["ethics_score"] = self.ethics.get_overall_ethics_score()
        
        return status
