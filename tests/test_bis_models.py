from python_anesthesia_simulator.pd_models import BIS_model

# BIS_model object to test the Bouillon propofol-remifentanil interaction model
bouillon_bis_model = BIS_model('Bouillon')
#bouillon_bis_model.plot_surface()

# BIS_model object to test the Vanluchene propofol model
vanluchene_bis_model = BIS_model('Vanluchene')
#vanluchene_bis_model.plot_surface()

# Create a BIS_model object initialized with custom parmeters equal to those of bouillon
custom_bis_model_bouillon = BIS_model(hill_param = [4.47, 19.3, 1.43, 0, 97.4, 97.4])

# Create a BIS_model object initialized with custom parmeters equal to those of vanluchene
custom_bis_model_vanluchene = BIS_model(hill_param = [4.92, 0, 2.69, 0, 95.9, 87.5])

# tests
def test_default_initialization():
    """Ensure that the default models give correct results"""
    # Check results at low concentrations
    assert vanluchene_bis_model.compute_bis(0) >= 90
    assert bouillon_bis_model.compute_bis(0,0) >= 90
    # Check results at high concentrations
    assert vanluchene_bis_model.compute_bis(16) <= 20
    assert bouillon_bis_model.compute_bis(12,8) <= 20
    # Check results at clinically recommended concentrations
    assert vanluchene_bis_model.compute_bis(5) <= 60
    assert vanluchene_bis_model.compute_bis(5) >= 40
    assert bouillon_bis_model.compute_bis(3,6) <= 60
    assert bouillon_bis_model.compute_bis(3,6) >= 40
    
def test_custom_initialization():
    """Ensure that the custom models give correct results by co mparing with 
    the results given by the default models"""
    # Check results at low concentrations
    assert abs(vanluchene_bis_model.compute_bis(0) - 
               custom_bis_model_vanluchene.compute_bis(0)) < 1e-3
    assert abs(bouillon_bis_model.compute_bis(0,0) - 
               custom_bis_model_bouillon.compute_bis(0,0)) < 1e-3
    # Check results at high concentrations
    assert abs(vanluchene_bis_model.compute_bis(16) - 
               custom_bis_model_vanluchene.compute_bis(16)) < 1e-3
    assert abs(bouillon_bis_model.compute_bis(12,8) - 
               custom_bis_model_bouillon.compute_bis(12,8)) < 1e-3
    # Check results at clinically recommended concentrations
    assert abs(vanluchene_bis_model.compute_bis(5) - 
               custom_bis_model_vanluchene.compute_bis(5)) < 1e-3
    assert abs(bouillon_bis_model.compute_bis(3,6) - 
               custom_bis_model_bouillon.compute_bis(3,6)) < 1e-3
    
def test_inverse_hill():
    """Check that the inversion of the hill function is giving correct results"""
    # Check results at low concentrations
    assert abs(vanluchene_bis_model.inverse_hill
               (vanluchene_bis_model.compute_bis(0)) - 0) < 1e-3
    assert abs(bouillon_bis_model.inverse_hill
               (bouillon_bis_model.compute_bis(0,0),0) - 0) < 1e-3
    # Check results at high concentrations
    assert abs(vanluchene_bis_model.inverse_hill
               (vanluchene_bis_model.compute_bis(16)) - 16) < 1e-3
    assert abs(bouillon_bis_model.inverse_hill
               (bouillon_bis_model.compute_bis(12,8),8) - 12) < 1e-3
    # Check results at clinically recommended concentrations
    assert abs(vanluchene_bis_model.inverse_hill
               (vanluchene_bis_model.compute_bis(5)) - 5) < 1e-3
    assert abs(bouillon_bis_model.inverse_hill
               (bouillon_bis_model.compute_bis(3,6),6) - 3) < 1e-3