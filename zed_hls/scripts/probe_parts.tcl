set fp [open "installed_parts.log" w]
puts $fp [get_parts]
close $fp
exit
