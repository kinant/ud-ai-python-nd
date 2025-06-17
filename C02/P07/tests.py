def run_tests(shirt_one, shirt_two, total, total_discounted):
 
    # Unit tests to check your solution
    assert shirt_one.price == 10, 'shirt_one price should be 10'
    assert shirt_one.color == 'red', ' shirt_one should be red'
    assert shirt_one.style == 'long-sleeve', 'shirt_one should be long_sleeve style'
    assert shirt_one.size == 'S', 'shirt_one size should be S'

    assert shirt_two.price == 10, 'shirt_two price should be 10'
    assert shirt_two.color == 'orange', 'shirt_two should be orange'
    assert shirt_two.style == 'short-sleeve', 'shirt_two should be short_sleeve style'
    assert shirt_two.size == 'L', 'shirt_two size should be L'

    assert total == 20, 'the total of both shirts should be 20'
    
    assert round(total_discounted) == 18, 'total_discounted should be 18.0'